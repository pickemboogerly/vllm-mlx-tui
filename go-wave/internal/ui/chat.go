package ui

import (
	"context"
	"fmt"
	"go-wave/internal/client"
	"go-wave/internal/server"
	"strings"
	"time"

	"github.com/atotto/clipboard"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/lipgloss"
)

// L3: single source of truth for the metrics pane width
const metricsPaneWidth = 30

type Theme struct {
	UserBg   lipgloss.Color
	UserFg   lipgloss.Color
	AIBg     lipgloss.Color
	AIFg     lipgloss.Color
	Border   lipgloss.Color
	Footer   lipgloss.Color
	HeaderBg lipgloss.Color
	HeaderFg lipgloss.Color
}

var themes = []Theme{
	{ // Ocean
		UserBg:   lipgloss.Color("32"),
		UserFg:   lipgloss.Color("255"),
		AIBg:     lipgloss.Color("24"),
		AIFg:     lipgloss.Color("255"),
		Border:   lipgloss.Color("39"),
		Footer:   lipgloss.Color("38"),
		HeaderBg: lipgloss.Color("39"),
		HeaderFg: lipgloss.Color("232"),
	},
	{ // Cyberpunk
		UserBg:   lipgloss.Color("205"),
		UserFg:   lipgloss.Color("0"),
		AIBg:     lipgloss.Color("51"),
		AIFg:     lipgloss.Color("0"),
		Border:   lipgloss.Color("205"),
		Footer:   lipgloss.Color("129"),
		HeaderBg: lipgloss.Color("226"),
		HeaderFg: lipgloss.Color("0"),
	},
	{ // Monochrome
		UserBg:   lipgloss.Color("250"),
		UserFg:   lipgloss.Color("232"),
		AIBg:     lipgloss.Color("238"),
		AIFg:     lipgloss.Color("255"),
		Border:   lipgloss.Color("244"),
		Footer:   lipgloss.Color("244"),
		HeaderBg: lipgloss.Color("255"),
		HeaderFg: lipgloss.Color("232"),
	},
	{ // Dracula
		UserBg:   lipgloss.Color("#bd93f9"),
		UserFg:   lipgloss.Color("#282a36"),
		AIBg:     lipgloss.Color("#44475a"),
		AIFg:     lipgloss.Color("#f8f8f2"),
		Border:   lipgloss.Color("#6272a4"),
		Footer:   lipgloss.Color("#6272a4"),
		HeaderBg: lipgloss.Color("#ff79c6"),
		HeaderFg: lipgloss.Color("#282a36"),
	},
	{ // Gruvbox
		UserBg:   lipgloss.Color("#d3869b"),
		UserFg:   lipgloss.Color("#282828"),
		AIBg:     lipgloss.Color("#3c3836"),
		AIFg:     lipgloss.Color("#ebdbb2"),
		Border:   lipgloss.Color("#a89984"),
		Footer:   lipgloss.Color("#928374"),
		HeaderBg: lipgloss.Color("#b8bb26"),
		HeaderFg: lipgloss.Color("#282828"),
	},
	{ // Synthwave
		UserBg:   lipgloss.Color("#f92aad"),
		UserFg:   lipgloss.Color("#ffffff"),
		AIBg:     lipgloss.Color("#2a2139"),
		AIFg:     lipgloss.Color("#f0eff1"),
		Border:   lipgloss.Color("#f92aad"),
		Footer:   lipgloss.Color("#34294f"),
		HeaderBg: lipgloss.Color("#20FBBD"),
		HeaderFg: lipgloss.Color("#2a2139"),
	},
}

type Msg struct {
	content string
	isUser  bool
}

type ChatModel struct {
	modelID   string
	manager   *server.Manager
	textInput textinput.Model
	viewport  viewport.Model
	spinner   spinner.Model

	messages []Msg

	generating bool
	cancelGen  context.CancelFunc
	sub        <-chan string
	metricsMsg string

	width     int
	height    int
	tokens    int
	startTime time.Time

	themeIdx int

	// H4: cached glamour renderer — re-created only when width changes,
	// not on every streaming token.
	mdRenderer      *glamour.TermRenderer
	mdRendererWidth int

	// L1: cached lipgloss bubble styles — re-created only when theme or
	// viewport width changes, not on every render call.
	cachedUserStyle lipgloss.Style
	cachedAIStyle   lipgloss.Style
	styleThemeIdx   int
	styleWidth      int
	stylesValid     bool
}

func NewChatModel(modelID string, manager *server.Manager) *ChatModel {
	ti := textinput.New()
	ti.Placeholder = "Type a message... (Enter to send)"
	ti.Focus()
	ti.CharLimit = 1000

	vp := viewport.New(0, 0)
	vp.YPosition = 0

	sp := spinner.New()
	sp.Spinner = spinner.Dot

	return &ChatModel{
		modelID:    modelID,
		manager:    manager,
		textInput:  ti,
		viewport:   vp,
		spinner:    sp,
		messages:   []Msg{{content: fmt.Sprintf("SYSTEM: Connected to %s", modelID), isUser: false}},
		metricsMsg: "Ready.",
		themeIdx:   0,
	}
}

type chunkMsg string
type doneMsg struct{}
type tickMsg time.Time
type clearMetricsMsg struct{}

func ticker() tea.Cmd {
	return tea.Tick(time.Second*1, func(t time.Time) tea.Msg {
		return tickMsg(t)
	})
}

func clearMetricsCmd() tea.Cmd {
	return tea.Tick(time.Second*3, func(_ time.Time) tea.Msg {
		return clearMetricsMsg{}
	})
}

// startStream starts a streaming request. The returned channel is owned by the
// goroutine inside StreamChat (on success) or by this function's wrapper goroutine
// (on error). Either way it is closed exactly once (C1).
func startStream(ctx context.Context, modelID, prompt string, port int) (<-chan string, tea.Cmd) {
	// Buffered by 1 so that an error message can be sent without blocking
	// even if the receiver isn't ready yet.
	chunks := make(chan string, 1)
	go func() {
		err := client.StreamChat(ctx, port, modelID, prompt, chunks)
		if err != nil {
			// StreamChat returned an error before launching its goroutine,
			// so we own the channel and must close it (C1).
			select {
			case chunks <- fmt.Sprintf("\n[Connection Error: %v]", err):
			case <-ctx.Done():
			}
			close(chunks)
		}
		// On nil return, StreamChat's goroutine owns and will close chunks.
	}()
	return chunks, waitForActivity(chunks)
}

func waitForActivity(sub <-chan string) tea.Cmd {
	return func() tea.Msg {
		chunk, ok := <-sub
		if !ok {
			return doneMsg{}
		}
		return chunkMsg(chunk)
	}
}

func (m *ChatModel) Init() tea.Cmd {
	return tea.Batch(textinput.Blink, ticker(), m.spinner.Tick)
}

// renderMarkdown renders markdown with a cached TermRenderer (H4).
// The renderer is reused while width stays constant; re-created on width change.
func (m *ChatModel) renderMarkdown(input string, width int) string {
	if width < 10 {
		width = 80
	}
	// H4: reuse renderer if width hasn't changed
	if m.mdRenderer == nil || m.mdRendererWidth != width {
		r, err := glamour.NewTermRenderer(
			glamour.WithStandardStyle("dark"),
			glamour.WithWordWrap(width),
		)
		if err != nil {
			return input
		}
		m.mdRenderer = r
		m.mdRendererWidth = width
	}
	out, err := m.mdRenderer.Render(input)
	if err != nil {
		return input
	}
	// glamour adds a leading newline; trim it for cleaner bubble layout
	return strings.TrimLeft(out, "\n")
}

func (m *ChatModel) renderMessages() string {
	var b strings.Builder

	width := m.width - 2
	if width < 20 {
		width = 20
	}
	maxBubbleWidth := int(float64(width) * 0.75)

	// L1: recompute styles only when theme or width changes, not every render
	if !m.stylesValid || m.styleThemeIdx != m.themeIdx || m.styleWidth != width {
		theme := themes[m.themeIdx]
		m.cachedUserStyle = lipgloss.NewStyle().
			Background(theme.UserBg).
			Foreground(theme.UserFg).
			Padding(0, 1).
			Border(lipgloss.RoundedBorder()).
			BorderForeground(theme.Border).
			Width(maxBubbleWidth)
		m.cachedAIStyle = lipgloss.NewStyle().
			Background(theme.AIBg).
			Foreground(theme.AIFg).
			Padding(0, 1).
			Border(lipgloss.RoundedBorder()).
			BorderForeground(theme.Border).
			Width(maxBubbleWidth)
		m.styleThemeIdx = m.themeIdx
		m.styleWidth = width
		m.stylesValid = true
	}
	userStyle := m.cachedUserStyle
	aiStyle := m.cachedAIStyle

	for _, msg := range m.messages {
		if msg.isUser {
			// User messages: plain text, right-aligned bubble
			content := lipgloss.NewStyle().Width(maxBubbleWidth).Render(msg.content)
			bubble := userStyle.Render(content)
			bubbleWidth := lipgloss.Width(bubble)
			padding := width - bubbleWidth
			if padding < 0 {
				padding = 0
			}
			spacer := strings.Repeat(" ", padding)
			b.WriteString(lipgloss.JoinHorizontal(lipgloss.Top, spacer, bubble) + "\n\n")
		} else {
			// AI messages: render markdown with glamour then wrap in bubble
			formatted := m.renderMarkdown(msg.content, maxBubbleWidth-4)
			bubble := aiStyle.Render(formatted)
			b.WriteString(bubble + "\n\n")
		}
	}
	return b.String()
}

func (m *ChatModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd
	var cmds []tea.Cmd

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height

		// L3: use metricsPaneWidth constant (+ 2 for border) to derive chat width
		chatWidth := m.width - (metricsPaneWidth + 2)
		if chatWidth < 20 {
			chatWidth = m.width // Fallback if terminal too narrow
		}

		m.textInput.Width = chatWidth - 4
		if m.textInput.Width < 0 {
			m.textInput.Width = 0
		}

		m.viewport.Width = chatWidth - 2
		if m.viewport.Width < 0 {
			m.viewport.Width = 0
		}

		m.viewport.Height = m.height - 7
		if m.viewport.Height < 0 {
			m.viewport.Height = 0
		}

		// Invalidate style cache on resize (L1)
		m.stylesValid = false

		m.viewport.SetContent(m.renderMessages())
		m.viewport.GotoBottom()

	case tickMsg:
		cmds = append(cmds, ticker())

	case clearMetricsMsg:
		m.metricsMsg = ""

	case tea.KeyMsg:
		if msg.String() == "ctrl+y" {
			if !m.generating && len(m.messages) > 0 {
				for i := len(m.messages) - 1; i >= 0; i-- {
					if !m.messages[i].isUser {
						_ = clipboard.WriteAll(m.messages[i].content)
						m.metricsMsg = "Copied last AI message to clipboard!"
						return m, clearMetricsCmd()
					}
				}
			}
			return m, nil
		}
		if msg.String() == "ctrl+t" {
			m.themeIdx = (m.themeIdx + 1) % len(themes)
			m.stylesValid = false // L1: invalidate style cache on theme change
			m.viewport.SetContent(m.renderMessages())
			return m, nil
		}

		if msg.Type == tea.KeyEnter && !m.generating {
			prompt := m.textInput.Value()
			if prompt != "" {
				m.messages = append(m.messages, Msg{content: prompt, isUser: true})
				m.messages = append(m.messages, Msg{content: "", isUser: false})
				m.textInput.SetValue("")
				m.generating = true
				m.tokens = 0
				m.startTime = time.Now()

				m.viewport.SetContent(m.renderMessages())
				m.viewport.GotoBottom()

				// H2: cancel any in-flight generation before starting a new one
				if m.cancelGen != nil {
					m.cancelGen()
				}

				ctx, cancel := context.WithCancel(context.Background())
				m.cancelGen = cancel
				port := 8000
				if m.manager != nil {
					port = m.manager.Port
				}

				sub, subCmd := startStream(ctx, m.modelID, prompt, port)
				m.sub = sub
				cmds = append(cmds, subCmd)
			}
		}

	case chunkMsg:
		m.tokens++
		m.messages[len(m.messages)-1].content += string(msg)
		m.viewport.SetContent(m.renderMessages())
		m.viewport.GotoBottom()
		return m, waitForActivity(m.sub)

	case doneMsg:
		m.generating = false
	}

	m.textInput, cmd = m.textInput.Update(msg)
	cmds = append(cmds, cmd)

	m.viewport, cmd = m.viewport.Update(msg)
	cmds = append(cmds, cmd)

	if m.generating {
		m.spinner, cmd = m.spinner.Update(msg)
		cmds = append(cmds, cmd)
	}

	return m, tea.Batch(cmds...)
}

func (m *ChatModel) View() string {
	theme := themes[m.themeIdx]

	headerStyle := lipgloss.NewStyle().
		Background(theme.HeaderBg).
		Foreground(theme.HeaderFg).
		Padding(0, 1).
		Bold(true).
		Width(m.width)
	headerStr := headerStyle.Render("🌊 VLLMWaveApp")

	// Chat Area
	vpStr := m.viewport.View()

	var statusStr string
	if m.generating {
		statusStr = lipgloss.NewStyle().Foreground(theme.Footer).Render(m.spinner.View() + " generating ...")
	} else {
		statusStr = lipgloss.NewStyle().Foreground(theme.Footer).Render("")
	}

	inputStr := m.textInput.View()
	chatArea := lipgloss.JoinVertical(lipgloss.Left, vpStr, statusStr, inputStr)

	// Metrics Pane
	metricsPane := m.renderMetricsPane()

	// Horizontal Layout
	var mainLayout string
	if m.width > 50 {
		mainLayout = lipgloss.JoinHorizontal(lipgloss.Top, chatArea, metricsPane)
	} else {
		mainLayout = chatArea
	}

	// Footer
	footerLabelStyle := lipgloss.NewStyle().
		Background(theme.AIBg).
		Foreground(theme.Footer).
		Width(m.width).
		Align(lipgloss.Center)

	footerLabel := footerLabelStyle.Render(fmt.Sprintf("Model: %s | Text Select: Option-Drag (Mac) / Shift-Drag (PC)", m.modelID))

	footerHelpText := "Commands: [Enter] Send • [Ctrl+Y] Copy AI Msg • [Ctrl+T] Theme • [Ctrl+C] Quit"
	if m.metricsMsg != "" && m.metricsMsg != "Ready." {
		footerHelpText += " | " + m.metricsMsg
	}

	footerHelpStyle := lipgloss.NewStyle().Foreground(theme.Footer)
	footerHelp := footerHelpStyle.Render(footerHelpText)

	return lipgloss.JoinVertical(lipgloss.Left, headerStr, mainLayout, footerLabel, footerHelp)
}

func (m *ChatModel) renderMetricsPane() string {
	theme := themes[m.themeIdx]

	// L3: use metricsPaneWidth constant
	metricsStyle := lipgloss.NewStyle().
		Width(metricsPaneWidth).
		Border(lipgloss.NormalBorder(), false, false, false, true).
		BorderForeground(lipgloss.Color("42")).
		PaddingLeft(1).
		Height(m.height - 4)

	labelStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("205")).Bold(true).PaddingTop(1).PaddingBottom(1)
	valStyle := lipgloss.NewStyle().Foreground(theme.AIFg)

	memVal := "0.0 MB"
	tpsVal := "0.0 t/s"

	if m.manager != nil {
		mem := m.manager.MemoryMB()
		memVal = fmt.Sprintf("%.1f MB", mem)
		if m.generating {
			elapsed := time.Since(m.startTime).Seconds()
			if elapsed > 0 {
				tpsVal = fmt.Sprintf("%.1f t/s", float64(m.tokens)/elapsed)
			}
		}
	}

	s := labelStyle.Render("Memory Usage") + "\n"
	s += valStyle.Render(memVal) + "\n"
	s += labelStyle.Render("Generation Speed") + "\n"
	s += valStyle.Render(tpsVal) + "\n"

	return metricsStyle.Render(s)
}
