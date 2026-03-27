package ui

import (
	"fmt"
	"go-wave/internal/cache"
	"go-wave/internal/server"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

type RouterState int

const (
	StateLauncher RouterState = iota
	StateBooting
	StateChat
)

type AppModel struct {
	state         RouterState
	models        []string
	cursor        int
	manager       *server.Manager
	selectedModel string
	errorMsg      string

	chatModel *ChatModel
	width     int
	height    int
}

func NewAppModel() *AppModel {
	return &AppModel{
		state:  StateLauncher,
		models: cache.ListCachedModels(),
	}
}

func (m *AppModel) Init() tea.Cmd {
	return nil
}

type BootMsg struct {
	success bool
	err     string
}

func doBoot(model string, manager *server.Manager) tea.Cmd {
	return func() tea.Msg {
		success, stderr := manager.Start()
		return BootMsg{success: success, err: stderr}
	}
}

func (m *AppModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
	case tea.KeyMsg:
		if msg.String() == "ctrl+c" {
			// H2: cancel any in-flight generation before quitting
			if m.chatModel != nil && m.chatModel.cancelGen != nil {
				m.chatModel.cancelGen()
			}
			if m.manager != nil {
				m.manager.Stop()
			}
			return m, tea.Quit
		}
	case BootMsg:
		if msg.success {
			m.state = StateChat
			m.chatModel = NewChatModel(m.selectedModel, m.manager)
			_, _ = m.chatModel.Update(tea.WindowSizeMsg{Width: m.width, Height: m.height})
			return m, m.chatModel.Init()
		} else {
			m.state = StateLauncher
			m.errorMsg = msg.err
			m.manager.Stop()
			m.manager = nil
		}
	}

	switch m.state {
	case StateLauncher:
		return m.updateLauncher(msg)
	case StateBooting:
		return m, nil
	case StateChat:
		var cmd tea.Cmd
		cm, cmd := m.chatModel.Update(msg)
		m.chatModel = cm.(*ChatModel)
		return m, cmd
	}

	return m, nil
}

func (m *AppModel) updateLauncher(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "up", "k":
			if m.cursor > 0 {
				m.cursor--
			}
		case "down", "j":
			if m.cursor < len(m.models)-1 {
				m.cursor++
			}
		// M6: q and esc were advertised in help text but not handled
		case "q", "esc":
			if m.manager != nil {
				m.manager.Stop()
			}
			return m, tea.Quit
		case "enter":
			// M2: guard against empty model list (no longer has fallback models)
			if len(m.models) == 0 {
				return m, nil
			}
			m.state = StateBooting
			m.selectedModel = m.models[m.cursor]
			m.manager = server.NewManager(m.selectedModel)
			return m, doBoot(m.selectedModel, m.manager)
		}
	}
	return m, nil
}

// view code
var (
	titleStyle        = lipgloss.NewStyle().Foreground(lipgloss.Color("205")).Bold(true).MarginTop(1).MarginBottom(1)
	itemStyle         = lipgloss.NewStyle().PaddingLeft(2)
	selectedItemStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("170")).Bold(true).PaddingLeft(2).Border(lipgloss.NormalBorder(), false, false, false, true).BorderForeground(lipgloss.Color("205"))
	errorStyle        = lipgloss.NewStyle().Foreground(lipgloss.Color("196")).MarginTop(2)
)

func (m *AppModel) View() string {
	switch m.state {
	case StateLauncher:
		s := titleStyle.Render("🌊 VLLMWaveApp") + "\n\n"

		if len(m.models) == 0 {
			// L6: no fake fallback models — show a clear empty-state message
			s += errorStyle.Render("No models found in ~/.cache/huggingface/hub") + "\n"
			s += lipgloss.NewStyle().PaddingLeft(2).Render("Download a model first, then restart this app.") + "\n"
		} else {
			s += lipgloss.NewStyle().PaddingLeft(2).Render("Select a model from your HuggingFace cache:") + "\n\n"
			for i, choice := range m.models {
				if i == m.cursor {
					s += selectedItemStyle.Render("[*] "+choice) + "\n"
				} else {
					s += itemStyle.Render("[ ] "+choice) + "\n"
				}
			}
			s += "\n" + lipgloss.NewStyle().PaddingLeft(2).Foreground(lipgloss.Color("42")).Render("[Launch Model]") + "\n"
		}

		if m.errorMsg != "" {
			s += errorStyle.Render("Failed to Boot: "+m.errorMsg) + "\n"
		}
		s += "\n\n(Press enter to launch, go up/down with arrows/j/k, q/esc/ctrl+c to quit)"
		return s
	case StateBooting:
		return titleStyle.Render(fmt.Sprintf("Booting %s ... this could take a few moments.", m.selectedModel))
	case StateChat:
		return m.chatModel.View()
	}
	return ""
}
