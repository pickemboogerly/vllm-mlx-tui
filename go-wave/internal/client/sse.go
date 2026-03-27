package client

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

type SSEMessage struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
	} `json:"choices"`
}

// StreamChat connects to the vllm-mlx SSE endpoint and streams chunks onto the
// provided channel. On error it returns immediately (channel untouched). On
// success it starts a goroutine that owns and closes the channel when done.
// The goroutine respects ctx cancellation (C1).
func StreamChat(ctx context.Context, port int, modelID, prompt string, chunks chan<- string) error {
	payload := map[string]interface{}{
		"model": modelID,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
		"stream": true,
	}

	// L2: handle json.Marshal error explicitly
	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", port), bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		var p []byte
		p, _ = io.ReadAll(resp.Body)
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(p))
	}

	// C1: goroutine owns the channel; defer close ensures it is always closed
	// exactly once. A select on ctx.Done() inside the loop prevents the
	// goroutine from leaking when the caller cancels the context.
	go func() {
		defer resp.Body.Close()
		defer close(chunks)

		// M3: 1 MB buffer prevents silent truncation of large SSE lines
		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 1<<20), 1<<20)

		scanner.Split(func(data []byte, atEOF bool) (int, []byte, error) {
			if atEOF && len(data) == 0 {
				return 0, nil, nil
			}
			if i := bytes.Index(data, []byte("\n\n")); i >= 0 {
				return i + 2, data[0:i], nil
			}
			if atEOF {
				return len(data), data, nil
			}
			return 0, nil, nil
		})

		for scanner.Scan() {
			// C1: bail out immediately on context cancellation
			select {
			case <-ctx.Done():
				return
			default:
			}

			text := scanner.Text()
			if len(text) > 6 && text[:6] == "data: " {
				content := text[6:]
				if content == "[DONE]" {
					break
				}

				var msg SSEMessage
				if err := json.Unmarshal([]byte(content), &msg); err == nil {
					if len(msg.Choices) > 0 {
						// C1: use select so a cancelled context doesn't block
						select {
						case chunks <- msg.Choices[0].Delta.Content:
						case <-ctx.Done():
							return
						}
					}
				}
			}
		}
	}()

	return nil
}
