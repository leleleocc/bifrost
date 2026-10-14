package passthroughparse

import (
	"bytes"
	"mime/multipart"
	"net/textproto"
	"strings"
	"testing"

	"github.com/bytedance/sonic"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/require"
)

// multipartBody builds a multipart/form-data body with text fields and one file part per file.
func multipartBody(t testing.TB, fields [][2]string, files map[string][]byte) (string, []byte) {
	t.Helper()
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	for _, kv := range fields {
		require.NoError(t, w.WriteField(kv[0], kv[1]))
	}
	for field, data := range files {
		header := textproto.MIMEHeader{}
		header.Set("Content-Disposition", `form-data; name="`+field+`"; filename="upload-`+strings.TrimSuffix(field, "[]")+`.bin"`)
		header.Set("Content-Type", "application/octet-stream")
		part, err := w.CreatePart(header)
		require.NoError(t, err)
		_, err = part.Write(data)
		require.NoError(t, err)
	}
	require.NoError(t, w.Close())
	return w.FormDataContentType(), buf.Bytes()
}

func TestSummarizeMultipart(t *testing.T) {
	audio := bytes.Repeat([]byte{0x00, 0xff, 0x10}, 1000)
	contentType, body := multipartBody(t, [][2]string{
		{"model", "whisper-1"}, {"language", "en"},
		{"timestamp_granularities[]", "word"}, {"timestamp_granularities[]", "segment"},
	}, map[string][]byte{"file": audio})

	summary, ok := SummarizeMultipart(contentType, body)
	require.True(t, ok)
	require.False(t, bytes.Contains(summary, audio[:30]), "file bytes must never be kept")

	var decoded MultipartSummary
	require.NoError(t, sonic.Unmarshal(summary, &decoded))
	require.Equal(t, "whisper-1", decoded.Fields["model"])
	require.Equal(t, []any{"word", "segment"}, decoded.Fields["timestamp_granularities[]"])
	require.Len(t, decoded.Files, 1)
	require.Equal(t, MultipartFile{Field: "file", Filename: "upload-file.bin", ContentType: "application/octet-stream", Bytes: int64(len(audio))}, decoded.Files[0])

	// Byte-stable: the same body always summarizes to the same JSON.
	again, _ := SummarizeMultipart(contentType, body)
	require.Equal(t, summary, again)

	_, ok = SummarizeMultipart("application/json", body)
	require.False(t, ok)
	_, ok = SummarizeMultipart("multipart/form-data", body)
	require.False(t, ok, "no boundary")
	_, ok = SummarizeMultipart(contentType, body[:len(body)/2])
	require.False(t, ok, "truncated body")
}

func TestParseMultipartRequests(t *testing.T) {
	summarize := func(fields [][2]string, files map[string][]byte) []byte {
		ct, body := multipartBody(t, fields, files)
		summary, ok := SummarizeMultipart(ct, body)
		require.True(t, ok)
		return summary
	}

	t.Run("openai transcription", func(t *testing.T) {
		body := summarize([][2]string{{"model", "whisper-1"}, {"language", "en"}, {"temperature", "0.2"}, {"response_format", "json"}, {"timestamp_granularities[]", "word"}}, map[string][]byte{"file": []byte("RIFF....")})
		r, err := Parse(Input{Provider: schemas.OpenAI, Model: "whisper-1", Method: "POST", Path: "/v1/audio/transcriptions", StatusCode: 200, RequestBody: body, ResponseBody: []byte(`{"text":"hello"}`)})
		require.NoError(t, err)
		req := r.Request.TranscriptionRequest
		require.NotNil(t, req)
		require.Equal(t, "upload-file.bin", req.Input.Filename)
		require.Empty(t, req.Input.File)
		require.Equal(t, "en", *req.Params.Language)
		require.InDelta(t, 0.2, *req.Params.Temperature, 1e-9)
		require.Equal(t, []string{"word"}, req.Params.TimestampGranularities)
		require.Equal(t, "hello", r.Response.TranscriptionResponse.Text)
	})
	t.Run("openai image edit", func(t *testing.T) {
		body := summarize([][2]string{{"model", "gpt-image-1"}, {"prompt", "add a hat"}, {"n", "2"}, {"size", "1024x1024"}}, map[string][]byte{"image[]": []byte("png")})
		r, err := Parse(Input{Provider: schemas.OpenAI, Model: "gpt-image-1", Method: "POST", Path: "/v1/images/edits", StatusCode: 200, RequestBody: body, ResponseBody: []byte(`{"created":1,"data":[{"b64_json":"aGk="}]}`)})
		require.NoError(t, err)
		req := r.Request.ImageEditRequest
		require.NotNil(t, req)
		require.Equal(t, "add a hat", req.Input.Prompt)
		require.Len(t, req.Input.Images, 1)
		require.Equal(t, 2, *req.Params.N)
		require.Len(t, r.Response.ImageGenerationResponse.Data, 1)
	})
	t.Run("openai image variation", func(t *testing.T) {
		body := summarize([][2]string{{"model", "dall-e-2"}, {"n", "1"}, {"size", "256x256"}}, map[string][]byte{"image": []byte("png")})
		r, err := Parse(Input{Provider: schemas.OpenAI, Model: "dall-e-2", Method: "POST", Path: "/v1/images/variations", StatusCode: 200, RequestBody: body})
		require.NoError(t, err)
		require.NotNil(t, r.Request.ImageVariationRequest)
		require.Equal(t, "256x256", *r.Request.ImageVariationRequest.Params.Size)
	})
	t.Run("a raw multipart body is not a summary", func(t *testing.T) {
		_, raw := multipartBody(t, [][2]string{{"model", "whisper-1"}}, map[string][]byte{"file": []byte("x")})
		r, err := Parse(Input{Provider: schemas.OpenAI, Model: "whisper-1", Method: "POST", Path: "/v1/audio/transcriptions", StatusCode: 200, RequestBody: raw})
		require.NoError(t, err)
		require.Nil(t, r)
	})
}

// TestSummarizeMultipartIsBestEffort pins that malformed bodies are rejected, never a panic.
func TestSummarizeMultipartIsBestEffort(t *testing.T) {
	ct := "multipart/form-data; boundary=XYZ"
	for _, body := range []string{
		"", "--XYZ", "--XYZ--", "--XYZ\r\n", "--XYZ\r\nContent-Disposition: form-data\r\n\r\nvalue",
		"--XYZ\r\nbroken header\r\n\r\nv\r\n--XYZ--", "--XYZ\r\nContent-Disposition: ;;;\r\n\r\nv\r\n--XYZ--",
		"preamble\r\n--XYZ\r\nContent-Disposition: form-data; name=\"a\"\r\n\r\n1\r\n--XYZ--",
	} {
		require.NotPanics(t, func() { _, _ = SummarizeMultipart(ct, []byte(body)) }, "body %q", body)
	}
	summary, ok := SummarizeMultipart(ct, []byte("preamble\r\n--XYZ\r\nContent-Disposition: form-data; name=\"a\"\r\n\r\n1\r\n--XYZ--"))
	require.True(t, ok)
	require.JSONEq(t, `{"fields":{"a":"1"}}`, string(summary))
}

// BenchmarkSummarizeMultipart records the request-path cost of logging a multipart passthrough
// upload. It runs in PreLLMHook only when content logging stores content, and scans the body once.
func BenchmarkSummarizeMultipart(b *testing.B) {
	contentType, body := multipartBody(b, [][2]string{{"model", "whisper-1"}, {"language", "en"}}, map[string][]byte{"file": bytes.Repeat([]byte{0xab}, 1<<20)})
	b.SetBytes(int64(len(body)))
	b.ReportAllocs()
	for b.Loop() {
		_, _ = SummarizeMultipart(contentType, body)
	}
}
