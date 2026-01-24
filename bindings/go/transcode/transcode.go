// Package transcode provides Go bindings for the transcode library.
//
// This package wraps the transcode C API to provide idiomatic Go access
// to video transcoding functionality.
//
// Example usage:
//
//	ctx, err := transcode.OpenInput("input.mp4")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer ctx.Close()
//
//	for {
//	    pkt, err := ctx.ReadPacket()
//	    if err == transcode.ErrEndOfStream {
//	        break
//	    }
//	    // Process packet...
//	}
package transcode

/*
#cgo CFLAGS: -I${SRCDIR}/../../../transcode-capi
#cgo LDFLAGS: -L${SRCDIR}/../../../target/release -ltranscode_capi

#include "transcode.h"
#include <stdlib.h>
*/
import "C"
import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

// Error codes
var (
	ErrInvalidArgument   = errors.New("transcode: invalid argument")
	ErrNullPointer       = errors.New("transcode: null pointer")
	ErrEndOfStream       = errors.New("transcode: end of stream")
	ErrIO                = errors.New("transcode: I/O error")
	ErrCodec             = errors.New("transcode: codec error")
	ErrContainer         = errors.New("transcode: container error")
	ErrResourceExhausted = errors.New("transcode: resource exhausted")
	ErrUnsupported       = errors.New("transcode: unsupported feature")
	ErrCancelled         = errors.New("transcode: cancelled")
	ErrBufferTooSmall    = errors.New("transcode: buffer too small")
	ErrInvalidState      = errors.New("transcode: invalid state")
	ErrUnknown           = errors.New("transcode: unknown error")
)

// StreamType represents the type of media stream.
type StreamType int

const (
	StreamTypeUnknown  StreamType = C.TRANSCODE_STREAM_TYPE_UNKNOWN
	StreamTypeVideo    StreamType = C.TRANSCODE_STREAM_TYPE_VIDEO
	StreamTypeAudio    StreamType = C.TRANSCODE_STREAM_TYPE_AUDIO
	StreamTypeSubtitle StreamType = C.TRANSCODE_STREAM_TYPE_SUBTITLE
	StreamTypeData     StreamType = C.TRANSCODE_STREAM_TYPE_DATA
)

// String returns a human-readable name for the stream type.
func (st StreamType) String() string {
	switch st {
	case StreamTypeVideo:
		return "video"
	case StreamTypeAudio:
		return "audio"
	case StreamTypeSubtitle:
		return "subtitle"
	case StreamTypeData:
		return "data"
	default:
		return "unknown"
	}
}

// PixelFormat represents the pixel format of video frames.
type PixelFormat int

const (
	PixelFormatUnknown     PixelFormat = C.TRANSCODE_PIXEL_FORMAT_UNKNOWN
	PixelFormatYUV420P     PixelFormat = C.TRANSCODE_PIXEL_FORMAT_YUV420P
	PixelFormatYUV422P     PixelFormat = C.TRANSCODE_PIXEL_FORMAT_YUV422P
	PixelFormatYUV444P     PixelFormat = C.TRANSCODE_PIXEL_FORMAT_YUV444P
	PixelFormatYUV420P10LE PixelFormat = C.TRANSCODE_PIXEL_FORMAT_YUV420P10LE
	PixelFormatYUV422P10LE PixelFormat = C.TRANSCODE_PIXEL_FORMAT_YUV422P10LE
	PixelFormatYUV444P10LE PixelFormat = C.TRANSCODE_PIXEL_FORMAT_YUV444P10LE
	PixelFormatNV12        PixelFormat = C.TRANSCODE_PIXEL_FORMAT_NV12
	PixelFormatNV21        PixelFormat = C.TRANSCODE_PIXEL_FORMAT_NV21
	PixelFormatRGB24       PixelFormat = C.TRANSCODE_PIXEL_FORMAT_RGB24
	PixelFormatBGR24       PixelFormat = C.TRANSCODE_PIXEL_FORMAT_BGR24
	PixelFormatRGBA        PixelFormat = C.TRANSCODE_PIXEL_FORMAT_RGBA
	PixelFormatBGRA        PixelFormat = C.TRANSCODE_PIXEL_FORMAT_BGRA
	PixelFormatGray8       PixelFormat = C.TRANSCODE_PIXEL_FORMAT_GRAY8
	PixelFormatGray16      PixelFormat = C.TRANSCODE_PIXEL_FORMAT_GRAY16
)

// ColorSpace represents the color space of video frames.
type ColorSpace int

const (
	ColorSpaceBT601  ColorSpace = C.TRANSCODE_COLOR_SPACE_BT601
	ColorSpaceBT709  ColorSpace = C.TRANSCODE_COLOR_SPACE_BT709
	ColorSpaceBT2020 ColorSpace = C.TRANSCODE_COLOR_SPACE_BT2020
	ColorSpaceSRGB   ColorSpace = C.TRANSCODE_COLOR_SPACE_SRGB
)

// ColorRange represents the color range of video frames.
type ColorRange int

const (
	ColorRangeLimited ColorRange = C.TRANSCODE_COLOR_RANGE_LIMITED
	ColorRangeFull    ColorRange = C.TRANSCODE_COLOR_RANGE_FULL
)

// PacketFlags for encoded packets.
const (
	PacketFlagKeyframe   = C.TRANSCODE_PACKET_FLAG_KEYFRAME
	PacketFlagCorrupt    = C.TRANSCODE_PACKET_FLAG_CORRUPT
	PacketFlagDiscard    = C.TRANSCODE_PACKET_FLAG_DISCARD
	PacketFlagDisposable = C.TRANSCODE_PACKET_FLAG_DISPOSABLE
)

// FrameFlags for decoded frames.
const (
	FrameFlagKeyframe      = C.TRANSCODE_FRAME_FLAG_KEYFRAME
	FrameFlagCorrupt       = C.TRANSCODE_FRAME_FLAG_CORRUPT
	FrameFlagDiscard       = C.TRANSCODE_FRAME_FLAG_DISCARD
	FrameFlagInterlaced    = C.TRANSCODE_FRAME_FLAG_INTERLACED
	FrameFlagTopFieldFirst = C.TRANSCODE_FRAME_FLAG_TOP_FIELD_FIRST
)

// Version returns the version string of the transcode library.
func Version() string {
	return C.GoString(C.transcode_version())
}

// errorFromCode converts a C error code to a Go error.
func errorFromCode(code C.enum_TranscodeError) error {
	switch code {
	case C.TRANSCODE_ERROR_SUCCESS:
		return nil
	case C.TRANSCODE_ERROR_INVALID_ARGUMENT:
		return ErrInvalidArgument
	case C.TRANSCODE_ERROR_NULL_POINTER:
		return ErrNullPointer
	case C.TRANSCODE_ERROR_END_OF_STREAM:
		return ErrEndOfStream
	case C.TRANSCODE_ERROR_IO_ERROR:
		return ErrIO
	case C.TRANSCODE_ERROR_CODEC_ERROR:
		return ErrCodec
	case C.TRANSCODE_ERROR_CONTAINER_ERROR:
		return ErrContainer
	case C.TRANSCODE_ERROR_RESOURCE_EXHAUSTED:
		return ErrResourceExhausted
	case C.TRANSCODE_ERROR_UNSUPPORTED:
		return ErrUnsupported
	case C.TRANSCODE_ERROR_CANCELLED:
		return ErrCancelled
	case C.TRANSCODE_ERROR_BUFFER_TOO_SMALL:
		return ErrBufferTooSmall
	case C.TRANSCODE_ERROR_INVALID_STATE:
		return ErrInvalidState
	default:
		return fmt.Errorf("%w: code %d", ErrUnknown, code)
	}
}

// StreamInfo contains information about a media stream.
type StreamInfo struct {
	Index         int
	Type          StreamType
	CodecID       uint32
	Width         int
	Height        int
	PixelFormat   PixelFormat
	SampleRate    int
	Channels      int
	BitsPerSample int
	TimeBaseNum   int
	TimeBaseDen   int
	Duration      int64
	Bitrate       uint64
}

// Context represents a transcode context for reading/writing media files.
type Context struct {
	ctx *C.struct_TranscodeContext
}

// OpenInput opens an input file for reading.
func OpenInput(path string) (*Context, error) {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	var cctx *C.struct_TranscodeContext
	err := C.transcode_open_input(cpath, &cctx)
	if err != C.TRANSCODE_ERROR_SUCCESS {
		return nil, errorFromCode(err)
	}

	ctx := &Context{ctx: cctx}
	runtime.SetFinalizer(ctx, (*Context).Close)
	return ctx, nil
}

// Close closes the context and frees resources.
func (c *Context) Close() error {
	if c.ctx != nil {
		C.transcode_close(c.ctx)
		c.ctx = nil
	}
	return nil
}

// NumStreams returns the number of streams in the input.
func (c *Context) NumStreams() int {
	if c.ctx == nil {
		return 0
	}
	return int(c.ctx.num_streams)
}

// StreamInfo returns information about a specific stream.
func (c *Context) StreamInfo(index int) (*StreamInfo, error) {
	if c.ctx == nil {
		return nil, ErrInvalidState
	}

	var cinfo C.struct_TranscodeStreamInfo
	err := C.transcode_get_stream_info(c.ctx, C.uint32_t(index), &cinfo)
	if err != C.TRANSCODE_ERROR_SUCCESS {
		return nil, errorFromCode(err)
	}

	return &StreamInfo{
		Index:         int(cinfo.index),
		Type:          StreamType(cinfo.stream_type),
		CodecID:       uint32(cinfo.codec_id),
		Width:         int(cinfo.width),
		Height:        int(cinfo.height),
		PixelFormat:   PixelFormat(cinfo.pixel_format),
		SampleRate:    int(cinfo.sample_rate),
		Channels:      int(cinfo.channels),
		BitsPerSample: int(cinfo.bits_per_sample),
		TimeBaseNum:   int(cinfo.time_base_num),
		TimeBaseDen:   int(cinfo.time_base_den),
		Duration:      int64(cinfo.duration),
		Bitrate:       uint64(cinfo.bitrate),
	}, nil
}

// Config holds encoding configuration options.
type Config struct {
	Width        int
	Height       int
	PixelFormat  PixelFormat
	Bitrate      uint64
	MaxBitrate   uint64
	Quality      int
	GOPSize      int
	BFrames      int
	FramerateNum int
	FramerateDen int
	SampleRate   int
	Channels     int
	Preset       int
	Threads      int
}

// OpenOutput opens an output file for writing.
func (c *Context) OpenOutput(path string, config *Config) error {
	if c.ctx == nil {
		return ErrInvalidState
	}

	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	var cconfig *C.struct_TranscodeConfig
	if config != nil {
		cfg := C.struct_TranscodeConfig{
			width:         C.uint32_t(config.Width),
			height:        C.uint32_t(config.Height),
			pixel_format:  C.enum_TranscodePixelFormat(config.PixelFormat),
			bitrate:       C.uint64_t(config.Bitrate),
			max_bitrate:   C.uint64_t(config.MaxBitrate),
			quality:       C.int32_t(config.Quality),
			gop_size:      C.uint32_t(config.GOPSize),
			b_frames:      C.uint32_t(config.BFrames),
			framerate_num: C.uint32_t(config.FramerateNum),
			framerate_den: C.uint32_t(config.FramerateDen),
			sample_rate:   C.uint32_t(config.SampleRate),
			channels:      C.uint32_t(config.Channels),
			preset:        C.uint32_t(config.Preset),
			threads:       C.uint32_t(config.Threads),
		}
		cconfig = &cfg
	}

	err := C.transcode_open_output(c.ctx, cpath, cconfig)
	if err != C.TRANSCODE_ERROR_SUCCESS {
		return errorFromCode(err)
	}
	return nil
}

// Packet represents an encoded media packet.
type Packet struct {
	pkt *C.struct_TranscodePacket
}

// NewPacket allocates a new packet.
func NewPacket() *Packet {
	pkt := C.transcode_packet_alloc()
	if pkt == nil {
		return nil
	}

	p := &Packet{pkt: pkt}
	runtime.SetFinalizer(p, (*Packet).Free)
	return p
}

// Free releases the packet resources.
func (p *Packet) Free() {
	if p.pkt != nil {
		C.transcode_packet_free(p.pkt)
		p.pkt = nil
	}
}

// Data returns the packet data as a byte slice.
func (p *Packet) Data() []byte {
	if p.pkt == nil || p.pkt.data == nil || p.pkt.size == 0 {
		return nil
	}
	return C.GoBytes(unsafe.Pointer(p.pkt.data), C.int(p.pkt.size))
}

// Size returns the packet data size.
func (p *Packet) Size() int {
	if p.pkt == nil {
		return 0
	}
	return int(p.pkt.size)
}

// PTS returns the presentation timestamp.
func (p *Packet) PTS() int64 {
	if p.pkt == nil {
		return 0
	}
	return int64(p.pkt.pts)
}

// DTS returns the decode timestamp.
func (p *Packet) DTS() int64 {
	if p.pkt == nil {
		return 0
	}
	return int64(p.pkt.dts)
}

// Duration returns the packet duration.
func (p *Packet) Duration() int64 {
	if p.pkt == nil {
		return 0
	}
	return int64(p.pkt.duration)
}

// StreamIndex returns the stream index.
func (p *Packet) StreamIndex() int {
	if p.pkt == nil {
		return 0
	}
	return int(p.pkt.stream_index)
}

// Flags returns the packet flags.
func (p *Packet) Flags() uint32 {
	if p.pkt == nil {
		return 0
	}
	return uint32(p.pkt.flags)
}

// IsKeyframe returns true if this is a keyframe packet.
func (p *Packet) IsKeyframe() bool {
	return p.Flags()&PacketFlagKeyframe != 0
}

// ReadPacket reads the next packet from the input.
func (c *Context) ReadPacket(pkt *Packet) error {
	if c.ctx == nil || pkt == nil || pkt.pkt == nil {
		return ErrInvalidArgument
	}

	err := C.transcode_read_packet(c.ctx, pkt.pkt)
	if err != C.TRANSCODE_ERROR_SUCCESS {
		return errorFromCode(err)
	}
	return nil
}

// WritePacket writes a packet to the output.
func (c *Context) WritePacket(pkt *Packet) error {
	if c.ctx == nil || pkt == nil || pkt.pkt == nil {
		return ErrInvalidArgument
	}

	err := C.transcode_write_packet(c.ctx, pkt.pkt)
	if err != C.TRANSCODE_ERROR_SUCCESS {
		return errorFromCode(err)
	}
	return nil
}

// Frame represents a decoded video frame.
type Frame struct {
	frm *C.struct_TranscodeFrame
}

// NewFrame allocates a new frame.
func NewFrame() *Frame {
	frm := C.transcode_frame_alloc()
	if frm == nil {
		return nil
	}

	f := &Frame{frm: frm}
	runtime.SetFinalizer(f, (*Frame).Free)
	return f
}

// Free releases the frame resources.
func (f *Frame) Free() {
	if f.frm != nil {
		C.transcode_frame_free(f.frm)
		f.frm = nil
	}
}

// AllocBuffer allocates a frame buffer with the specified dimensions.
func (f *Frame) AllocBuffer(width, height int, format PixelFormat) error {
	if f.frm == nil {
		return ErrInvalidState
	}

	err := C.transcode_frame_alloc_buffer(f.frm, C.uint32_t(width), C.uint32_t(height), C.enum_TranscodePixelFormat(format))
	if err != C.TRANSCODE_ERROR_SUCCESS {
		return errorFromCode(err)
	}
	return nil
}

// Width returns the frame width.
func (f *Frame) Width() int {
	if f.frm == nil {
		return 0
	}
	return int(f.frm.width)
}

// Height returns the frame height.
func (f *Frame) Height() int {
	if f.frm == nil {
		return 0
	}
	return int(f.frm.height)
}

// Format returns the pixel format.
func (f *Frame) Format() PixelFormat {
	if f.frm == nil {
		return PixelFormatUnknown
	}
	return PixelFormat(f.frm.format)
}

// PTS returns the presentation timestamp.
func (f *Frame) PTS() int64 {
	if f.frm == nil {
		return 0
	}
	return int64(f.frm.pts)
}

// DTS returns the decode timestamp.
func (f *Frame) DTS() int64 {
	if f.frm == nil {
		return 0
	}
	return int64(f.frm.dts)
}

// Flags returns the frame flags.
func (f *Frame) Flags() uint32 {
	if f.frm == nil {
		return 0
	}
	return uint32(f.frm.flags)
}

// IsKeyframe returns true if this is a keyframe.
func (f *Frame) IsKeyframe() bool {
	return f.Flags()&FrameFlagKeyframe != 0
}

// DecodePacket decodes a packet into a frame.
func (c *Context) DecodePacket(pkt *Packet, frm *Frame) error {
	if c.ctx == nil {
		return ErrInvalidState
	}

	var cpkt *C.struct_TranscodePacket
	if pkt != nil {
		cpkt = pkt.pkt
	}

	var cfrm *C.struct_TranscodeFrame
	if frm != nil {
		cfrm = frm.frm
	}

	err := C.transcode_decode_packet(c.ctx, cpkt, cfrm)
	if err != C.TRANSCODE_ERROR_SUCCESS {
		return errorFromCode(err)
	}
	return nil
}

// EncodeFrame encodes a frame into a packet.
func (c *Context) EncodeFrame(frm *Frame, pkt *Packet) error {
	if c.ctx == nil {
		return ErrInvalidState
	}

	var cfrm *C.struct_TranscodeFrame
	if frm != nil {
		cfrm = frm.frm
	}

	var cpkt *C.struct_TranscodePacket
	if pkt != nil {
		cpkt = pkt.pkt
	}

	err := C.transcode_encode_frame(c.ctx, cfrm, cpkt)
	if err != C.TRANSCODE_ERROR_SUCCESS {
		return errorFromCode(err)
	}
	return nil
}

// Seek seeks to a specific timestamp.
func (c *Context) Seek(streamIndex int, timestamp int64) error {
	if c.ctx == nil {
		return ErrInvalidState
	}

	err := C.transcode_seek(c.ctx, C.int(streamIndex), C.int64_t(timestamp), 0)
	if err != C.TRANSCODE_ERROR_SUCCESS {
		return errorFromCode(err)
	}
	return nil
}

// FlushDecoder flushes the decoder buffers.
func (c *Context) FlushDecoder() error {
	if c.ctx == nil {
		return ErrInvalidState
	}

	err := C.transcode_flush_decoder(c.ctx)
	if err != C.TRANSCODE_ERROR_SUCCESS {
		return errorFromCode(err)
	}
	return nil
}

// FlushEncoder flushes the encoder buffers.
func (c *Context) FlushEncoder() error {
	if c.ctx == nil {
		return ErrInvalidState
	}

	err := C.transcode_flush_encoder(c.ctx)
	if err != C.TRANSCODE_ERROR_SUCCESS {
		return errorFromCode(err)
	}
	return nil
}
