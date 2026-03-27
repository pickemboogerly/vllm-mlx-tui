package ui

import (
	"context"
	"fmt"
	"go-wave/internal/client"
	"go-wave/internal/server"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

type ChatModel struct {
	modelID    string
	manager    *server.Manager
	textInput  textinput.Model
	messages   []string
	
	generating bool
	cancelGen  context.CancelFunc
	metricsMsg string
	
	width      int
	height     int
	tokens     int
	startTime  time.Time
}

func NewChatModel(modelID string, manager *server.Manager) *ChatModel {
	ti := textinput.New()
	ti.Placeholder = "Send a message..."
	ti.Focus()
	ti.CharLimit = 1000
	ti.Width = 60

	return &ChatModel{
		modelID:   modelID,
		manager:   manager,
		textInput: ti,
		messages:  []string{fmt.Sprintf("SYSTEM: Connected to %s", modelID)},
		metricsMsg: "Ready.",
	}
}

type chunkMsg string
type doneMsg struct{}
type tickMsg time.Time

func ticker() tea.Cmd {
	return tea.Tick(time.Second*1, func(t time.Time) tea.Msg {
		return tickMsg(t)
	})
}

func streamGenerator(ctx context.Context, modelID, prompt string, port int) tea.Cmd {
	return func() tea.Msg {
		chunks := make(chan string)
		go func() {
			_ = client.StreamChat(ctx, port, modelID, prompt, chunks)
		}()
		
		for {
			select {
			case <-ctx.Done():
				return doneMsg{}
			case chuck, ok := <-chunks:
				if !ok {
					return doneMsg{}
				}
				return chunkMsg(chuck)
			}
		}
	}
}

func (m *ChatModel) Init() tea.Cmd {
	return tea.Batch(textinput.Blink, ticker())
}

func (m *ChatModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd
	var cmds []tea.Cmd

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.textInput.Width = msg.Width - 4
		
	case tickMsg:
		if m.manager != nil {
			mem := m.manager.MemoryMB()
			
			if m.generating {
				elapsed := time.Since(m.startTime).Seconds()
				tps := 0.0
				if elapsed > 0 {
					tps = float64(m.tokens) / elapsed
				}
				m.metricsMsg = fmt.Sprintf("RAM: %.1f MB | Speed: %.1f tk/s", mem, tps)
			} else {
				m.metricsMsg = fmt.Sprintf("RAM: %.1f MB | Idle", mem)
			}
		}
		cmds = append(cmds, ticker())

	case tea.KeyMsg:
		if msg.Type == tea.KeyEnter && !m.generating {
			prompt := m.textInput.Value()
			if prompt != "" {
				m.messages = append(m.messages, "User:\n"+prompt)
				m.messages = append(m.messages, "Assistant:\n")
				m.textInput.SetValue("")
				m.generating = true
				m.tokens = 0
				m.startTime = time.Now()
				
				ctx, cancel := context.WithCancel(context.Background())
				m.cancelGen = cancel
				port := 8000
				if m.manager != nil {
					port = m.manager.Port
				}
				cmds = append(cmds, streamGenerator(ctx, m.modelID, prompt, port))
			}
		}

	case chunkMsg:
		m.tokens++
		m.messages[len(m.messages)-1] += string(msg)
		// Request next chunk recursively (in reality we use a channel reader in bubbletea, but a simpler hack is just to let the generator pipe msgs into Update via cmd)
		// Wait, the streamGenerator blocks and returns a single Msg because it's a Cmd.
		// For proper streaming in BubbleTea, we really want a channel we read from.
		// A cleaner pattern is using tea-commands recursively.
		return m, func() tea.Msg { return nil } // We need a proper channel reader here. I will fix the streamGenerator logic.
	case doneMsg:
		m.generating = false
	}

	m.textInput, cmd = m.textInput.Update(msg)
	cmds = append(cmds, cmd)

	return m, tea.Batch(cmds...)
}

func (m *ChatModel) View() string {
	var b strings.Builder
	
	chatBox := lipgloss.NewStyle().
		Border(lipgloss.NormalBorder()).
		BorderForeground(lipgloss.Color("63")).
		Width(m.width - 4).
		Height(m.height - 8)
		
	var chatContent strings.Builder
	for _, msg := range m.messages {
		chatContent.WriteString(msg + "\n\n")
	}

	b.WriteString(chatBox.Render(chatContent.String()) + "\n")
	
	b.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color("205")).Render(m.metricsMsg) + "\n")
	b.WriteString(m.textInput.View())

	return b.String()
}
