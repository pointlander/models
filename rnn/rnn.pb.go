// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.23.0
// 	protoc        v3.11.4
// source: rnn.proto

package main

import (
	proto "github.com/golang/protobuf/proto"
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

// This is a compile-time assertion that a sufficiently up-to-date version
// of the legacy proto package is being used.
const _ = proto.ProtoPackageIsVersion4

type Vector struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Verse  uint64   `protobuf:"varint,1,opt,name=verse,proto3" json:"verse,omitempty"`
	Vector []uint32 `protobuf:"varint,2,rep,packed,name=vector,proto3" json:"vector,omitempty"`
}

func (x *Vector) Reset() {
	*x = Vector{}
	if protoimpl.UnsafeEnabled {
		mi := &file_rnn_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Vector) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Vector) ProtoMessage() {}

func (x *Vector) ProtoReflect() protoreflect.Message {
	mi := &file_rnn_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Vector.ProtoReflect.Descriptor instead.
func (*Vector) Descriptor() ([]byte, []int) {
	return file_rnn_proto_rawDescGZIP(), []int{0}
}

func (x *Vector) GetVerse() uint64 {
	if x != nil {
		return x.Verse
	}
	return 0
}

func (x *Vector) GetVector() []uint32 {
	if x != nil {
		return x.Vector
	}
	return nil
}

var File_rnn_proto protoreflect.FileDescriptor

var file_rnn_proto_rawDesc = []byte{
	0x0a, 0x09, 0x72, 0x6e, 0x6e, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x04, 0x6d, 0x61, 0x69,
	0x6e, 0x22, 0x36, 0x0a, 0x06, 0x56, 0x65, 0x63, 0x74, 0x6f, 0x72, 0x12, 0x14, 0x0a, 0x05, 0x76,
	0x65, 0x72, 0x73, 0x65, 0x18, 0x01, 0x20, 0x01, 0x28, 0x04, 0x52, 0x05, 0x76, 0x65, 0x72, 0x73,
	0x65, 0x12, 0x16, 0x0a, 0x06, 0x76, 0x65, 0x63, 0x74, 0x6f, 0x72, 0x18, 0x02, 0x20, 0x03, 0x28,
	0x0d, 0x52, 0x06, 0x76, 0x65, 0x63, 0x74, 0x6f, 0x72, 0x42, 0x08, 0x5a, 0x06, 0x2e, 0x3b, 0x6d,
	0x61, 0x69, 0x6e, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_rnn_proto_rawDescOnce sync.Once
	file_rnn_proto_rawDescData = file_rnn_proto_rawDesc
)

func file_rnn_proto_rawDescGZIP() []byte {
	file_rnn_proto_rawDescOnce.Do(func() {
		file_rnn_proto_rawDescData = protoimpl.X.CompressGZIP(file_rnn_proto_rawDescData)
	})
	return file_rnn_proto_rawDescData
}

var file_rnn_proto_msgTypes = make([]protoimpl.MessageInfo, 1)
var file_rnn_proto_goTypes = []interface{}{
	(*Vector)(nil), // 0: main.Vector
}
var file_rnn_proto_depIdxs = []int32{
	0, // [0:0] is the sub-list for method output_type
	0, // [0:0] is the sub-list for method input_type
	0, // [0:0] is the sub-list for extension type_name
	0, // [0:0] is the sub-list for extension extendee
	0, // [0:0] is the sub-list for field type_name
}

func init() { file_rnn_proto_init() }
func file_rnn_proto_init() {
	if File_rnn_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_rnn_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Vector); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_rnn_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   1,
			NumExtensions: 0,
			NumServices:   0,
		},
		GoTypes:           file_rnn_proto_goTypes,
		DependencyIndexes: file_rnn_proto_depIdxs,
		MessageInfos:      file_rnn_proto_msgTypes,
	}.Build()
	File_rnn_proto = out.File
	file_rnn_proto_rawDesc = nil
	file_rnn_proto_goTypes = nil
	file_rnn_proto_depIdxs = nil
}
