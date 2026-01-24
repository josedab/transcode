package transcode

import (
	"testing"
)

func TestVersion(t *testing.T) {
	v := Version()
	if v == "" {
		t.Error("Version() returned empty string")
	}
	t.Logf("Transcode version: %s", v)
}

func TestStreamTypeString(t *testing.T) {
	tests := []struct {
		st   StreamType
		want string
	}{
		{StreamTypeUnknown, "unknown"},
		{StreamTypeVideo, "video"},
		{StreamTypeAudio, "audio"},
		{StreamTypeSubtitle, "subtitle"},
		{StreamTypeData, "data"},
	}

	for _, tt := range tests {
		got := tt.st.String()
		if got != tt.want {
			t.Errorf("StreamType(%d).String() = %q, want %q", tt.st, got, tt.want)
		}
	}
}

func TestNewPacket(t *testing.T) {
	pkt := NewPacket()
	if pkt == nil {
		t.Fatal("NewPacket() returned nil")
	}
	defer pkt.Free()

	if pkt.Size() != 0 {
		t.Errorf("NewPacket().Size() = %d, want 0", pkt.Size())
	}

	if pkt.Data() != nil {
		t.Error("NewPacket().Data() != nil, want nil")
	}
}

func TestNewFrame(t *testing.T) {
	frm := NewFrame()
	if frm == nil {
		t.Fatal("NewFrame() returned nil")
	}
	defer frm.Free()

	if frm.Width() != 0 {
		t.Errorf("NewFrame().Width() = %d, want 0", frm.Width())
	}

	if frm.Height() != 0 {
		t.Errorf("NewFrame().Height() = %d, want 0", frm.Height())
	}

	if frm.Format() != PixelFormatUnknown {
		t.Errorf("NewFrame().Format() = %d, want %d", frm.Format(), PixelFormatUnknown)
	}
}

func TestOpenInputNotFound(t *testing.T) {
	_, err := OpenInput("/nonexistent/path/to/file.mp4")
	if err == nil {
		t.Error("OpenInput() with nonexistent file should return error")
	}
}

func TestPacketFlags(t *testing.T) {
	pkt := NewPacket()
	if pkt == nil {
		t.Fatal("NewPacket() returned nil")
	}
	defer pkt.Free()

	// New packet should not be a keyframe
	if pkt.IsKeyframe() {
		t.Error("New packet should not be a keyframe")
	}

	// Check flag values are defined
	if PacketFlagKeyframe == 0 {
		t.Error("PacketFlagKeyframe should not be 0")
	}
	if PacketFlagCorrupt == 0 {
		t.Error("PacketFlagCorrupt should not be 0")
	}
}

func TestFrameFlags(t *testing.T) {
	frm := NewFrame()
	if frm == nil {
		t.Fatal("NewFrame() returned nil")
	}
	defer frm.Free()

	// New frame should not be a keyframe
	if frm.IsKeyframe() {
		t.Error("New frame should not be a keyframe")
	}

	// Check flag values are defined
	if FrameFlagKeyframe == 0 {
		t.Error("FrameFlagKeyframe should not be 0")
	}
	if FrameFlagInterlaced == 0 {
		t.Error("FrameFlagInterlaced should not be 0")
	}
}

func TestPixelFormatConstants(t *testing.T) {
	formats := []struct {
		name   string
		format PixelFormat
	}{
		{"Unknown", PixelFormatUnknown},
		{"YUV420P", PixelFormatYUV420P},
		{"YUV422P", PixelFormatYUV422P},
		{"YUV444P", PixelFormatYUV444P},
		{"NV12", PixelFormatNV12},
		{"RGB24", PixelFormatRGB24},
		{"RGBA", PixelFormatRGBA},
	}

	seen := make(map[PixelFormat]string)
	for _, f := range formats {
		if prev, ok := seen[f.format]; ok {
			t.Errorf("PixelFormat %s has same value as %s", f.name, prev)
		}
		seen[f.format] = f.name
	}
}

func TestColorSpaceConstants(t *testing.T) {
	spaces := []struct {
		name  string
		space ColorSpace
	}{
		{"BT601", ColorSpaceBT601},
		{"BT709", ColorSpaceBT709},
		{"BT2020", ColorSpaceBT2020},
		{"SRGB", ColorSpaceSRGB},
	}

	seen := make(map[ColorSpace]string)
	for _, s := range spaces {
		if prev, ok := seen[s.space]; ok {
			t.Errorf("ColorSpace %s has same value as %s", s.name, prev)
		}
		seen[s.space] = s.name
	}
}

func TestErrorTypes(t *testing.T) {
	errors := []error{
		ErrInvalidArgument,
		ErrNullPointer,
		ErrEndOfStream,
		ErrIO,
		ErrCodec,
		ErrContainer,
		ErrResourceExhausted,
		ErrUnsupported,
		ErrCancelled,
		ErrBufferTooSmall,
		ErrInvalidState,
		ErrUnknown,
	}

	for _, err := range errors {
		if err == nil {
			t.Error("Error constant should not be nil")
		}
		if err.Error() == "" {
			t.Error("Error message should not be empty")
		}
	}
}

func TestContextNilOperations(t *testing.T) {
	// Operations on nil context should return appropriate errors
	ctx := &Context{}

	if ctx.NumStreams() != 0 {
		t.Error("NumStreams on nil context should return 0")
	}

	_, err := ctx.StreamInfo(0)
	if err != ErrInvalidState {
		t.Errorf("StreamInfo on nil context should return ErrInvalidState, got %v", err)
	}

	err = ctx.OpenOutput("test.mp4", nil)
	if err != ErrInvalidState {
		t.Errorf("OpenOutput on nil context should return ErrInvalidState, got %v", err)
	}

	err = ctx.FlushDecoder()
	if err != ErrInvalidState {
		t.Errorf("FlushDecoder on nil context should return ErrInvalidState, got %v", err)
	}

	err = ctx.FlushEncoder()
	if err != ErrInvalidState {
		t.Errorf("FlushEncoder on nil context should return ErrInvalidState, got %v", err)
	}
}

func TestConfig(t *testing.T) {
	cfg := Config{
		Width:        1920,
		Height:       1080,
		PixelFormat:  PixelFormatYUV420P,
		Bitrate:      5000000,
		Quality:      23,
		GOPSize:      60,
		FramerateNum: 30,
		FramerateDen: 1,
		Preset:       5,
		Threads:      4,
	}

	if cfg.Width != 1920 {
		t.Errorf("Config.Width = %d, want 1920", cfg.Width)
	}
	if cfg.Height != 1080 {
		t.Errorf("Config.Height = %d, want 1080", cfg.Height)
	}
	if cfg.Bitrate != 5000000 {
		t.Errorf("Config.Bitrate = %d, want 5000000", cfg.Bitrate)
	}
}
