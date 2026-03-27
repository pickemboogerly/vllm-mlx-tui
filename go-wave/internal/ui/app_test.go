package ui

import (
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

func TestUIRendersWithoutPanic(t *testing.T) {
	app := NewAppModel()

	// Initial Launcher state
	_ = app.View()

	// Move down once
	app.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'j'}})
	_ = app.View()

	// Press Enter to Boot
	app.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'\r'}})

	// manually force it to booting just in case
	app.state = StateBooting
	_ = app.View()

	var m tea.Model

	// Mocking Boot Success
	m, _ = app.Update(BootMsg{success: true, err: ""})
	app = m.(*AppModel)
	if app.state != StateChat {
		t.Fatalf("Expected StateChat, got %v", app.state)
	}

	// Trigger WindowSizeMsg
	m, _ = app.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	app = m.(*AppModel)

	// Trigger ticking
	m, _ = app.Update(tickMsg(time.Now()))
	app = m.(*AppModel)

	// Send message
	msg := tea.KeyMsg{Type: tea.KeyEnter}
	app.chatModel.textInput.SetValue("Hello there!")
	m, _ = app.Update(msg)
	app = m.(*AppModel)

	// Show view after message sent
	viewOutput := app.View()

	// Add a test chunk to simulate receiving AI tokens
	m, _ = app.Update(chunkMsg("Hi, I am AI."))
	app = m.(*AppModel)

	// Manually inject ctrl+y mock
	mockKey := tea.KeyMsg{Type: tea.KeyCtrlY}
	app.chatModel.generating = false // Ensure not generating so we can copy
	m, _ = app.Update(mockKey)
	app = m.(*AppModel)

	finalView := app.View()

	// L5: use t.Log instead of fmt.Println so output is suppressed on pass
	t.Log("--- CHAT VIEW RENDER TEST ---")
	t.Log(viewOutput)
	t.Log("\n--- FINAL CHAT VIEW (w/ toast) ---")
	t.Log(finalView)
}
