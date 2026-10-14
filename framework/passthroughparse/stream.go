package passthroughparse

import (
	"bytes"
	"errors"
	"io"
	"strings"

	"github.com/bytedance/sonic"
	"github.com/maximhq/bifrost/core/providers/anthropic"
	"github.com/maximhq/bifrost/core/providers/gemini"
	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/framework/streaming"
)

var sseDoneMarker = []byte("[DONE]")

// parseStreamResponse rebuilds the final response from a recorded stream body: the raw SSE text
// the passthrough accumulator concatenated, or a JSON array for Gemini without alt=sse. Each event
// is decoded the way the provider's own stream loop decodes it, then the deltas are assembled with
// the same builders the live streaming accumulator uses, so a parsed passthrough stream reads
// exactly like a typed one.
func parseStreamResponse(format wireFormat, requestType schemas.RequestType, body []byte) (*schemas.BifrostResponse, error) {
	switch format {
	case wireOpenAI:
		switch requestType {
		case schemas.ChatCompletionRequest:
			return assembleOpenAIChatStream(body)
		case schemas.TextCompletionRequest:
			return assembleOpenAITextStream(body)
		case schemas.ResponsesRequest:
			return assembleResponsesStream(body, decodeOpenAIResponsesEvent)
		}
	case wireAnthropic:
		if requestType == schemas.ResponsesRequest {
			return assembleAnthropicStream(body)
		}
	case wireGemini:
		if requestType == schemas.ResponsesRequest {
			return assembleGeminiStream(body)
		}
	}
	return nil, nil
}

// forEachSSEEvent calls fn with every SSE event's data payload in order, skipping the [DONE]
// terminator. Comments and id/retry fields are dropped by the shared reader.
func forEachSSEEvent(body []byte, fn func(eventType string, data []byte) error) error {
	reader := providerUtils.GetSSEEventReader(nil, bytes.NewReader(body))
	for {
		eventType, data, err := reader.ReadEvent()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		if len(data) == 0 || bytes.Equal(bytes.TrimSpace(data), sseDoneMarker) {
			continue
		}
		if err := fn(eventType, data); err != nil {
			return err
		}
	}
}

func assembleOpenAIChatStream(body []byte) (*schemas.BifrostResponse, error) {
	var (
		chunks       []*streaming.ChatStreamChunk
		finishReason *string
		usage        *schemas.BifrostLLMUsage
		first        *schemas.BifrostChatResponse
	)
	err := forEachSSEEvent(body, func(_ string, data []byte) error {
		var event schemas.BifrostChatResponse
		if err := sonic.Unmarshal(data, &event); err != nil {
			return err
		}
		if first == nil {
			first = &event
		}
		if event.Usage != nil {
			usage = event.Usage
		}
		if len(event.Choices) == 0 {
			return nil
		}
		choice := event.Choices[0]
		if choice.FinishReason != nil {
			finishReason = choice.FinishReason
		}
		if choice.ChatStreamResponseChoice != nil && choice.ChatStreamResponseChoice.Delta != nil {
			chunks = append(chunks, &streaming.ChatStreamChunk{ChunkIndex: len(chunks), Delta: choice.ChatStreamResponseChoice.Delta})
		}
		return nil
	})
	if first == nil {
		return nil, errors.Join(errNoEvents, err)
	}
	message := streaming.BuildChatMessageFromChunks(chunks)
	return &schemas.BifrostResponse{ChatResponse: &schemas.BifrostChatResponse{
		ID:      first.ID,
		Object:  "chat.completion",
		Created: first.Created,
		Model:   first.Model,
		Choices: []schemas.BifrostResponseChoice{{
			Index:                       0,
			FinishReason:                finishReason,
			ChatNonStreamResponseChoice: &schemas.ChatNonStreamResponseChoice{Message: message},
		}},
		Usage: usage,
	}}, err
}

func assembleOpenAITextStream(body []byte) (*schemas.BifrostResponse, error) {
	var (
		text         strings.Builder
		finishReason *string
		usage        *schemas.BifrostLLMUsage
		first        *schemas.BifrostTextCompletionResponse
	)
	err := forEachSSEEvent(body, func(_ string, data []byte) error {
		var event schemas.BifrostTextCompletionResponse
		if err := sonic.Unmarshal(data, &event); err != nil {
			return err
		}
		if first == nil {
			first = &event
		}
		if event.Usage != nil {
			usage = event.Usage
		}
		if len(event.Choices) == 0 {
			return nil
		}
		choice := event.Choices[0]
		if choice.FinishReason != nil {
			finishReason = choice.FinishReason
		}
		if choice.TextCompletionResponseChoice != nil && choice.TextCompletionResponseChoice.Text != nil {
			text.WriteString(*choice.TextCompletionResponseChoice.Text)
		}
		return nil
	})
	if first == nil {
		return nil, errors.Join(errNoEvents, err)
	}
	completion := text.String()
	return &schemas.BifrostResponse{TextCompletionResponse: &schemas.BifrostTextCompletionResponse{
		ID:     first.ID,
		Object: "text_completion",
		Model:  first.Model,
		Choices: []schemas.BifrostResponseChoice{{
			Index:                        0,
			FinishReason:                 finishReason,
			TextCompletionResponseChoice: &schemas.TextCompletionResponseChoice{Text: &completion},
		}},
		Usage: usage,
	}}, err
}

var errNoEvents = errors.New("stream body carried no decodable events")

// responsesEventDecoder turns one recorded event into zero or more Responses stream events.
type responsesEventDecoder func(eventType string, data []byte) ([]*schemas.BifrostResponsesStreamResponse, error)

func decodeOpenAIResponsesEvent(_ string, data []byte) ([]*schemas.BifrostResponsesStreamResponse, error) {
	var event schemas.BifrostResponsesStreamResponse
	if err := sonic.Unmarshal(data, &event); err != nil {
		return nil, err
	}
	return []*schemas.BifrostResponsesStreamResponse{&event}, nil
}

// responsesAssembler collects Responses stream events into the chunks the accumulator's builder
// consumes, and keeps the terminal envelope (status, stop reason, usage) from the last event that
// carried one.
type responsesAssembler struct {
	chunks   []*streaming.ResponsesStreamChunk
	envelope *schemas.BifrostResponsesResponse
}

func (a *responsesAssembler) add(events []*schemas.BifrostResponsesStreamResponse) {
	for _, event := range events {
		if event == nil {
			continue
		}
		if event.Response != nil {
			a.envelope = event.Response
		}
		a.chunks = append(a.chunks, &streaming.ResponsesStreamChunk{ChunkIndex: len(a.chunks), StreamResponse: event})
	}
}

func (a *responsesAssembler) result() *schemas.BifrostResponse {
	if len(a.chunks) == 0 {
		return nil
	}
	output := streaming.BuildResponsesMessagesFromChunks(a.chunks)
	resp := &schemas.BifrostResponsesResponse{}
	if a.envelope != nil {
		envelope := *a.envelope
		resp = &envelope
		// A completed event that already carries the full output (OpenAI) is authoritative;
		// converters that emit a bare completed envelope get the assembled deltas.
		if len(resp.Output) > 0 {
			return &schemas.BifrostResponse{ResponsesResponse: resp}
		}
	}
	resp.Output = output
	return &schemas.BifrostResponse{ResponsesResponse: resp}
}

func assembleResponsesStream(body []byte, decode responsesEventDecoder) (*schemas.BifrostResponse, error) {
	var assembler responsesAssembler
	err := forEachSSEEvent(body, func(eventType string, data []byte) error {
		events, err := decode(eventType, data)
		if err != nil {
			return err
		}
		assembler.add(events)
		return nil
	})
	resp := assembler.result()
	if resp == nil {
		return nil, errors.Join(errNoEvents, err)
	}
	return resp, err
}

func assembleAnthropicStream(body []byte) (*schemas.BifrostResponse, error) {
	ctx := newConverterContext()
	state := anthropic.AcquireAnthropicResponsesStreamState()
	defer anthropic.ReleaseAnthropicResponsesStreamState(state)
	sequence := 0
	return assembleResponsesStream(body, func(_ string, data []byte) ([]*schemas.BifrostResponsesStreamResponse, error) {
		var event anthropic.AnthropicStreamEvent
		if err := sonic.Unmarshal(data, &event); err != nil {
			return nil, err
		}
		events, bifrostErr, _ := event.ToBifrostResponsesStream(ctx, sequence, state)
		if bifrostErr != nil {
			return events, errors.New(bifrostErr.Error.Message)
		}
		sequence += len(events)
		return events, nil
	})
}

func assembleGeminiStream(body []byte) (*schemas.BifrostResponse, error) {
	state := gemini.AcquireGeminiResponsesStreamState()
	defer gemini.ReleaseGeminiResponsesStreamState(state)
	sequence := 0
	convert := func(chunk *gemini.GenerateContentResponse) ([]*schemas.BifrostResponsesStreamResponse, error) {
		events, bifrostErr := chunk.ToBifrostResponsesStream(sequence, state)
		if bifrostErr != nil {
			return events, errors.New(bifrostErr.Error.Message)
		}
		sequence += len(events)
		return events, nil
	}

	// Without alt=sse, :streamGenerateContent returns one JSON array of response chunks.
	if trimmed := bytes.TrimSpace(body); len(trimmed) > 0 && trimmed[0] == '[' {
		var chunks []*gemini.GenerateContentResponse
		if err := sonic.Unmarshal(trimmed, &chunks); err != nil {
			return nil, err
		}
		var assembler responsesAssembler
		for _, chunk := range chunks {
			if chunk == nil {
				continue
			}
			events, err := convert(chunk)
			assembler.add(events)
			if err != nil {
				return assembler.result(), err
			}
		}
		if resp := assembler.result(); resp != nil {
			return resp, nil
		}
		return nil, errNoEvents
	}

	return assembleResponsesStream(body, func(_ string, data []byte) ([]*schemas.BifrostResponsesStreamResponse, error) {
		var chunk gemini.GenerateContentResponse
		if err := sonic.Unmarshal(data, &chunk); err != nil {
			return nil, err
		}
		return convert(&chunk)
	})
}
