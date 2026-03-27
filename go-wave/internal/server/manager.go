package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os/exec"
	"regexp"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/shirou/gopsutil/v3/process"
)

type Manager struct {
	modelID    string
	Port       int
	cmd        *exec.Cmd

	// H1: mu protects stderrTail against concurrent read/write
	mu         sync.Mutex
	stderrTail []string

	// M4: cache MemoryMB result to avoid O(N-processes) syscalls every second
	lastMemRSS  uint64
	lastMemTime time.Time
}

func NewManager(modelID string) *Manager {
	return &Manager{
		modelID:    modelID,
		stderrTail: make([]string, 0),
	}
}

// Start launches vllm-mlx and returns true if it successfully boots, or false + error.
func (m *Manager) Start() (bool, string) {
	if !regexp.MustCompile(`^[a-zA-Z0-9/._-]+$`).MatchString(m.modelID) || strings.HasPrefix(m.modelID, "-") {
		return false, "Invalid model ID format"
	}

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return false, "Failed to allocate port: " + err.Error()
	}
	m.Port = listener.Addr().(*net.TCPAddr).Port
	listener.Close()

	_, err = exec.LookPath("vllm-mlx")
	if err != nil {
		return false, "vllm-mlx executable not found in PATH"
	}

	m.cmd = exec.Command("vllm-mlx", "serve", m.modelID, "--port", fmt.Sprintf("%d", m.Port))

	// Create a new process group to prevent zombie GPU processes
	m.cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	stderr, err := m.cmd.StderrPipe()
	if err != nil {
		return false, "Failed to capture stderr"
	}

	if err := m.cmd.Start(); err != nil {
		return false, "Failed to start process: " + err.Error()
	}

	// Read stderr concurrently to avoid blocking
	go m.readStderr(stderr)

	done := make(chan error, 1)
	go func() {
		done <- m.cmd.Wait()
	}()

	// Poll HTTP readiness
	ready := make(chan string, 1)
	go func() {
		client := http.Client{Timeout: time.Second}
		for i := 0; i < 120; i++ {
			req, _ := http.NewRequestWithContext(context.Background(), "GET", fmt.Sprintf("http://127.0.0.1:%d/v1/models", m.Port), nil)
			resp, err := client.Do(req)
			if err == nil && resp.StatusCode == 200 {
				resp.Body.Close()
				ready <- "ok"
				return
			}
			if resp != nil {
				resp.Body.Close()
			}
			time.Sleep(1 * time.Second)
		}
		ready <- "timeout"
	}()

	select {
	case <-done:
		// Process crashed prematurely; read tail under lock (H1)
		m.mu.Lock()
		tail := strings.Join(m.stderrTail, "\n")
		m.mu.Unlock()
		return false, fmt.Sprintf("Process exited before ready:\n%s", tail)
	case res := <-ready:
		if res == "ok" {
			// C2: Best-effort TOCTOU mitigation — verify at least one model is
			// actually served before declaring success.
			if !m.validateModelLoaded() {
				return false, "Server responded but returned no models (possible port collision)"
			}
			return true, ""
		}
		return false, "Timeout waiting for HTTP readiness"
	}
}

// validateModelLoaded checks that the /v1/models endpoint returns at least one
// model entry, providing a lightweight guard against port-hijacking (C2).
func (m *Manager) validateModelLoaded() bool {
	type modelsResp struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	url := fmt.Sprintf("http://127.0.0.1:%d/v1/models", m.Port)
	req, err := http.NewRequestWithContext(context.Background(), "GET", url, nil)
	if err != nil {
		return false
	}
	c := http.Client{Timeout: 5 * time.Second}
	resp, err := c.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	var result modelsResp
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return false
	}
	return len(result.Data) > 0
}

// readStderr drains stderr line-by-line, keeping the last 20 lines (H1: mutex protected).
func (m *Manager) readStderr(rc io.ReadCloser) {
	buf := make([]byte, 1024)
	for {
		n, err := rc.Read(buf)
		if n > 0 {
			lines := strings.Split(string(buf[:n]), "\n")
			m.mu.Lock() // H1: protect concurrent append/slice
			for _, l := range lines {
				if len(strings.TrimSpace(l)) > 0 {
					m.stderrTail = append(m.stderrTail, l)
					if len(m.stderrTail) > 20 {
						m.stderrTail = m.stderrTail[1:]
					}
				}
			}
			m.mu.Unlock()
		}
		if err != nil {
			break
		}
	}
}

// Stop cleanly kills the entire process group.
func (m *Manager) Stop() {
	if m.cmd != nil && m.cmd.Process != nil {
		pgid, err := syscall.Getpgid(m.cmd.Process.Pid)
		if err == nil {
			_ = syscall.Kill(-pgid, syscall.SIGKILL) // Hard kill group
		}
	}
}

// MemoryMB returns the total RSS of the server process group in megabytes.
// Results are cached for 2 seconds to avoid O(N-processes) syscalls every tick (M4).
func (m *Manager) MemoryMB() float64 {
	if m.cmd == nil || m.cmd.Process == nil {
		return 0
	}
	// M4: return cached value if fresh enough
	if time.Since(m.lastMemTime) < 2*time.Second {
		return float64(m.lastMemRSS) / (1024 * 1024)
	}

	pgid, err := syscall.Getpgid(m.cmd.Process.Pid)
	if err != nil {
		return 0
	}

	procs, err := process.Processes()
	if err != nil {
		return 0
	}

	var totalRSS uint64
	for _, p := range procs {
		pGid, _ := syscall.Getpgid(int(p.Pid))
		if pGid == pgid {
			mem, _ := p.MemoryInfo()
			if mem != nil {
				totalRSS += mem.RSS
			}
		}
	}

	m.lastMemRSS = totalRSS
	m.lastMemTime = time.Now()
	return float64(totalRSS) / (1024 * 1024)
}
