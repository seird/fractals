unsigned char src_kernels_julia_cl[] = {
  0x5f, 0x5f, 0x6b, 0x65, 0x72, 0x6e, 0x65, 0x6c, 0x20, 0x76, 0x6f, 0x69,
  0x64, 0x0d, 0x0a, 0x6a, 0x75, 0x6c, 0x69, 0x61, 0x28, 0x5f, 0x5f, 0x67,
  0x6c, 0x6f, 0x62, 0x61, 0x6c, 0x20, 0x75, 0x63, 0x68, 0x61, 0x72, 0x20,
  0x2a, 0x20, 0x4d, 0x2c, 0x20, 0x63, 0x6f, 0x6e, 0x73, 0x74, 0x20, 0x69,
  0x6e, 0x74, 0x20, 0x77, 0x69, 0x64, 0x74, 0x68, 0x2c, 0x20, 0x63, 0x6f,
  0x6e, 0x73, 0x74, 0x20, 0x69, 0x6e, 0x74, 0x20, 0x68, 0x65, 0x69, 0x67,
  0x68, 0x74, 0x2c, 0x20, 0x5f, 0x5f, 0x67, 0x6c, 0x6f, 0x62, 0x61, 0x6c,
  0x20, 0x73, 0x74, 0x72, 0x75, 0x63, 0x74, 0x20, 0x46, 0x72, 0x61, 0x63,
  0x74, 0x61, 0x6c, 0x50, 0x72, 0x6f, 0x70, 0x65, 0x72, 0x74, 0x69, 0x65,
  0x73, 0x20, 0x2a, 0x20, 0x66, 0x70, 0x29, 0x0d, 0x0a, 0x7b, 0x0d, 0x0a,
  0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x75, 0x69, 0x6e, 0x74, 0x20, 0x78,
  0x20, 0x3d, 0x20, 0x67, 0x65, 0x74, 0x5f, 0x67, 0x6c, 0x6f, 0x62, 0x61,
  0x6c, 0x5f, 0x69, 0x64, 0x28, 0x30, 0x29, 0x3b, 0x0d, 0x0a, 0x20, 0x20,
  0x20, 0x20, 0x75, 0x69, 0x6e, 0x74, 0x20, 0x79, 0x20, 0x3d, 0x20, 0x67,
  0x65, 0x74, 0x5f, 0x67, 0x6c, 0x6f, 0x62, 0x61, 0x6c, 0x5f, 0x69, 0x64,
  0x28, 0x31, 0x29, 0x3b, 0x0d, 0x0a, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20,
  0x66, 0x6c, 0x6f, 0x61, 0x74, 0x20, 0x7a, 0x5f, 0x72, 0x65, 0x61, 0x6c,
  0x20, 0x3d, 0x20, 0x66, 0x70, 0x2d, 0x3e, 0x78, 0x5f, 0x73, 0x74, 0x61,
  0x72, 0x74, 0x20, 0x2b, 0x20, 0x28, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x29,
  0x78, 0x2f, 0x77, 0x69, 0x64, 0x74, 0x68, 0x20, 0x2a, 0x20, 0x28, 0x66,
  0x70, 0x2d, 0x3e, 0x78, 0x5f, 0x65, 0x6e, 0x64, 0x20, 0x2d, 0x20, 0x66,
  0x70, 0x2d, 0x3e, 0x78, 0x5f, 0x73, 0x74, 0x61, 0x72, 0x74, 0x29, 0x3b,
  0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x20,
  0x7a, 0x5f, 0x69, 0x6d, 0x61, 0x67, 0x20, 0x3d, 0x20, 0x66, 0x70, 0x2d,
  0x3e, 0x79, 0x5f, 0x73, 0x74, 0x61, 0x72, 0x74, 0x20, 0x2b, 0x20, 0x28,
  0x66, 0x6c, 0x6f, 0x61, 0x74, 0x29, 0x79, 0x2f, 0x68, 0x65, 0x69, 0x67,
  0x68, 0x74, 0x20, 0x2a, 0x20, 0x28, 0x66, 0x70, 0x2d, 0x3e, 0x79, 0x5f,
  0x65, 0x6e, 0x64, 0x20, 0x2d, 0x20, 0x66, 0x70, 0x2d, 0x3e, 0x79, 0x5f,
  0x73, 0x74, 0x61, 0x72, 0x74, 0x29, 0x3b, 0x0d, 0x0a, 0x0d, 0x0a, 0x20,
  0x20, 0x20, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x20, 0x72, 0x5f, 0x72,
  0x65, 0x61, 0x6c, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x66, 0x6c,
  0x6f, 0x61, 0x74, 0x20, 0x72, 0x5f, 0x69, 0x6d, 0x61, 0x67, 0x3b, 0x0d,
  0x0a, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x69, 0x6e, 0x74, 0x20, 0x69,
  0x20, 0x3d, 0x20, 0x30, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x66,
  0x6f, 0x72, 0x20, 0x28, 0x3b, 0x20, 0x69, 0x3c, 0x66, 0x70, 0x2d, 0x3e,
  0x6d, 0x61, 0x78, 0x5f, 0x69, 0x74, 0x65, 0x72, 0x61, 0x74, 0x69, 0x6f,
  0x6e, 0x73, 0x3b, 0x20, 0x2b, 0x2b, 0x69, 0x29, 0x20, 0x7b, 0x0d, 0x0a,
  0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x69, 0x66, 0x20, 0x28,
  0x66, 0x72, 0x61, 0x63, 0x74, 0x61, 0x6c, 0x5f, 0x65, 0x73, 0x63, 0x61,
  0x70, 0x65, 0x5f, 0x6d, 0x61, 0x67, 0x6e, 0x69, 0x74, 0x75, 0x64, 0x65,
  0x5f, 0x63, 0x68, 0x65, 0x63, 0x6b, 0x28, 0x7a, 0x5f, 0x72, 0x65, 0x61,
  0x6c, 0x2c, 0x20, 0x7a, 0x5f, 0x69, 0x6d, 0x61, 0x67, 0x2c, 0x20, 0x66,
  0x70, 0x2d, 0x3e, 0x52, 0x29, 0x29, 0x20, 0x7b, 0x0d, 0x0a, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x62, 0x72,
  0x65, 0x61, 0x6b, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x7d, 0x0d, 0x0a, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x73, 0x77, 0x69, 0x74, 0x63, 0x68, 0x20, 0x28, 0x66,
  0x70, 0x2d, 0x3e, 0x66, 0x72, 0x61, 0x63, 0x29, 0x20, 0x7b, 0x0d, 0x0a,
  0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
  0x63, 0x61, 0x73, 0x65, 0x20, 0x46, 0x43, 0x5f, 0x46, 0x52, 0x41, 0x43,
  0x5f, 0x5a, 0x32, 0x3a, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x66, 0x72,
  0x61, 0x63, 0x74, 0x61, 0x6c, 0x5f, 0x7a, 0x32, 0x28, 0x26, 0x72, 0x5f,
  0x72, 0x65, 0x61, 0x6c, 0x2c, 0x20, 0x26, 0x72, 0x5f, 0x69, 0x6d, 0x61,
  0x67, 0x2c, 0x20, 0x7a, 0x5f, 0x72, 0x65, 0x61, 0x6c, 0x2c, 0x20, 0x7a,
  0x5f, 0x69, 0x6d, 0x61, 0x67, 0x2c, 0x20, 0x66, 0x70, 0x2d, 0x3e, 0x63,
  0x5f, 0x72, 0x65, 0x61, 0x6c, 0x2c, 0x20, 0x66, 0x70, 0x2d, 0x3e, 0x63,
  0x5f, 0x69, 0x6d, 0x61, 0x67, 0x29, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x62, 0x72, 0x65, 0x61, 0x6b, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x63, 0x61, 0x73,
  0x65, 0x20, 0x46, 0x43, 0x5f, 0x46, 0x52, 0x41, 0x43, 0x5f, 0x5a, 0x33,
  0x3a, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x66, 0x72, 0x61, 0x63, 0x74,
  0x61, 0x6c, 0x5f, 0x7a, 0x33, 0x28, 0x26, 0x72, 0x5f, 0x72, 0x65, 0x61,
  0x6c, 0x2c, 0x20, 0x26, 0x72, 0x5f, 0x69, 0x6d, 0x61, 0x67, 0x2c, 0x20,
  0x7a, 0x5f, 0x72, 0x65, 0x61, 0x6c, 0x2c, 0x20, 0x7a, 0x5f, 0x69, 0x6d,
  0x61, 0x67, 0x2c, 0x20, 0x66, 0x70, 0x2d, 0x3e, 0x63, 0x5f, 0x72, 0x65,
  0x61, 0x6c, 0x2c, 0x20, 0x66, 0x70, 0x2d, 0x3e, 0x63, 0x5f, 0x69, 0x6d,
  0x61, 0x67, 0x29, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x62, 0x72,
  0x65, 0x61, 0x6b, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x63, 0x61, 0x73, 0x65, 0x20, 0x46,
  0x43, 0x5f, 0x46, 0x52, 0x41, 0x43, 0x5f, 0x5a, 0x34, 0x3a, 0x0d, 0x0a,
  0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x20, 0x66, 0x72, 0x61, 0x63, 0x74, 0x61, 0x6c, 0x5f,
  0x7a, 0x34, 0x28, 0x26, 0x72, 0x5f, 0x72, 0x65, 0x61, 0x6c, 0x2c, 0x20,
  0x26, 0x72, 0x5f, 0x69, 0x6d, 0x61, 0x67, 0x2c, 0x20, 0x7a, 0x5f, 0x72,
  0x65, 0x61, 0x6c, 0x2c, 0x20, 0x7a, 0x5f, 0x69, 0x6d, 0x61, 0x67, 0x2c,
  0x20, 0x66, 0x70, 0x2d, 0x3e, 0x63, 0x5f, 0x72, 0x65, 0x61, 0x6c, 0x2c,
  0x20, 0x66, 0x70, 0x2d, 0x3e, 0x63, 0x5f, 0x69, 0x6d, 0x61, 0x67, 0x29,
  0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x62, 0x72, 0x65, 0x61, 0x6b,
  0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x63, 0x61, 0x73, 0x65, 0x20, 0x46, 0x43, 0x5f, 0x46,
  0x52, 0x41, 0x43, 0x5f, 0x5a, 0x43, 0x4f, 0x4e, 0x4a, 0x32, 0x3a, 0x0d,
  0x0a, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x20, 0x20, 0x66, 0x72, 0x61, 0x63, 0x74, 0x61, 0x6c,
  0x5f, 0x7a, 0x63, 0x6f, 0x6e, 0x6a, 0x32, 0x28, 0x26, 0x72, 0x5f, 0x72,
  0x65, 0x61, 0x6c, 0x2c, 0x20, 0x26, 0x72, 0x5f, 0x69, 0x6d, 0x61, 0x67,
  0x2c, 0x20, 0x7a, 0x5f, 0x72, 0x65, 0x61, 0x6c, 0x2c, 0x20, 0x7a, 0x5f,
  0x69, 0x6d, 0x61, 0x67, 0x2c, 0x20, 0x66, 0x70, 0x2d, 0x3e, 0x63, 0x5f,
  0x72, 0x65, 0x61, 0x6c, 0x2c, 0x20, 0x66, 0x70, 0x2d, 0x3e, 0x63, 0x5f,
  0x69, 0x6d, 0x61, 0x67, 0x29, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
  0x62, 0x72, 0x65, 0x61, 0x6b, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x63, 0x61, 0x73, 0x65,
  0x20, 0x46, 0x43, 0x5f, 0x46, 0x52, 0x41, 0x43, 0x5f, 0x5a, 0x43, 0x4f,
  0x4e, 0x4a, 0x33, 0x3a, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x66, 0x72,
  0x61, 0x63, 0x74, 0x61, 0x6c, 0x5f, 0x7a, 0x63, 0x6f, 0x6e, 0x6a, 0x33,
  0x28, 0x26, 0x72, 0x5f, 0x72, 0x65, 0x61, 0x6c, 0x2c, 0x20, 0x26, 0x72,
  0x5f, 0x69, 0x6d, 0x61, 0x67, 0x2c, 0x20, 0x7a, 0x5f, 0x72, 0x65, 0x61,
  0x6c, 0x2c, 0x20, 0x7a, 0x5f, 0x69, 0x6d, 0x61, 0x67, 0x2c, 0x20, 0x66,
  0x70, 0x2d, 0x3e, 0x63, 0x5f, 0x72, 0x65, 0x61, 0x6c, 0x2c, 0x20, 0x66,
  0x70, 0x2d, 0x3e, 0x63, 0x5f, 0x69, 0x6d, 0x61, 0x67, 0x29, 0x3b, 0x0d,
  0x0a, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x20, 0x20, 0x62, 0x72, 0x65, 0x61, 0x6b, 0x3b, 0x0d,
  0x0a, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x63, 0x61, 0x73, 0x65, 0x20, 0x46, 0x43, 0x5f, 0x46, 0x52, 0x41,
  0x43, 0x5f, 0x5a, 0x43, 0x4f, 0x4e, 0x4a, 0x34, 0x3a, 0x0d, 0x0a, 0x20,
  0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x66, 0x72, 0x61, 0x63, 0x74, 0x61, 0x6c, 0x5f, 0x7a,
  0x63, 0x6f, 0x6e, 0x6a, 0x34, 0x28, 0x26, 0x72, 0x5f, 0x72, 0x65, 0x61,
  0x6c, 0x2c, 0x20, 0x26, 0x72, 0x5f, 0x69, 0x6d, 0x61, 0x67, 0x2c, 0x20,
  0x7a, 0x5f, 0x72, 0x65, 0x61, 0x6c, 0x2c, 0x20, 0x7a, 0x5f, 0x69, 0x6d,
  0x61, 0x67, 0x2c, 0x20, 0x66, 0x70, 0x2d, 0x3e, 0x63, 0x5f, 0x72, 0x65,
  0x61, 0x6c, 0x2c, 0x20, 0x66, 0x70, 0x2d, 0x3e, 0x63, 0x5f, 0x69, 0x6d,
  0x61, 0x67, 0x29, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x62, 0x72,
  0x65, 0x61, 0x6b, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x63, 0x61, 0x73, 0x65, 0x20, 0x46,
  0x43, 0x5f, 0x46, 0x52, 0x41, 0x43, 0x5f, 0x5a, 0x41, 0x42, 0x53, 0x32,
  0x3a, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x66, 0x72, 0x61, 0x63, 0x74,
  0x61, 0x6c, 0x5f, 0x7a, 0x61, 0x62, 0x73, 0x32, 0x28, 0x26, 0x72, 0x5f,
  0x72, 0x65, 0x61, 0x6c, 0x2c, 0x20, 0x26, 0x72, 0x5f, 0x69, 0x6d, 0x61,
  0x67, 0x2c, 0x20, 0x7a, 0x5f, 0x72, 0x65, 0x61, 0x6c, 0x2c, 0x20, 0x7a,
  0x5f, 0x69, 0x6d, 0x61, 0x67, 0x2c, 0x20, 0x66, 0x70, 0x2d, 0x3e, 0x63,
  0x5f, 0x72, 0x65, 0x61, 0x6c, 0x2c, 0x20, 0x66, 0x70, 0x2d, 0x3e, 0x63,
  0x5f, 0x69, 0x6d, 0x61, 0x67, 0x29, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x62, 0x72, 0x65, 0x61, 0x6b, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x63, 0x61, 0x73,
  0x65, 0x20, 0x46, 0x43, 0x5f, 0x46, 0x52, 0x41, 0x43, 0x5f, 0x5a, 0x41,
  0x42, 0x53, 0x33, 0x3a, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x66, 0x72,
  0x61, 0x63, 0x74, 0x61, 0x6c, 0x5f, 0x7a, 0x61, 0x62, 0x73, 0x33, 0x28,
  0x26, 0x72, 0x5f, 0x72, 0x65, 0x61, 0x6c, 0x2c, 0x20, 0x26, 0x72, 0x5f,
  0x69, 0x6d, 0x61, 0x67, 0x2c, 0x20, 0x7a, 0x5f, 0x72, 0x65, 0x61, 0x6c,
  0x2c, 0x20, 0x7a, 0x5f, 0x69, 0x6d, 0x61, 0x67, 0x2c, 0x20, 0x66, 0x70,
  0x2d, 0x3e, 0x63, 0x5f, 0x72, 0x65, 0x61, 0x6c, 0x2c, 0x20, 0x66, 0x70,
  0x2d, 0x3e, 0x63, 0x5f, 0x69, 0x6d, 0x61, 0x67, 0x29, 0x3b, 0x0d, 0x0a,
  0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x20, 0x62, 0x72, 0x65, 0x61, 0x6b, 0x3b, 0x0d, 0x0a,
  0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
  0x63, 0x61, 0x73, 0x65, 0x20, 0x46, 0x43, 0x5f, 0x46, 0x52, 0x41, 0x43,
  0x5f, 0x5a, 0x41, 0x42, 0x53, 0x34, 0x3a, 0x0d, 0x0a, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x66, 0x72, 0x61, 0x63, 0x74, 0x61, 0x6c, 0x5f, 0x7a, 0x61, 0x62,
  0x73, 0x34, 0x28, 0x26, 0x72, 0x5f, 0x72, 0x65, 0x61, 0x6c, 0x2c, 0x20,
  0x26, 0x72, 0x5f, 0x69, 0x6d, 0x61, 0x67, 0x2c, 0x20, 0x7a, 0x5f, 0x72,
  0x65, 0x61, 0x6c, 0x2c, 0x20, 0x7a, 0x5f, 0x69, 0x6d, 0x61, 0x67, 0x2c,
  0x20, 0x66, 0x70, 0x2d, 0x3e, 0x63, 0x5f, 0x72, 0x65, 0x61, 0x6c, 0x2c,
  0x20, 0x66, 0x70, 0x2d, 0x3e, 0x63, 0x5f, 0x69, 0x6d, 0x61, 0x67, 0x29,
  0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x62, 0x72, 0x65, 0x61, 0x6b,
  0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x63, 0x61, 0x73, 0x65, 0x20, 0x46, 0x43, 0x5f, 0x46,
  0x52, 0x41, 0x43, 0x5f, 0x4d, 0x41, 0x47, 0x4e, 0x45, 0x54, 0x3a, 0x0d,
  0x0a, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x20, 0x20, 0x66, 0x72, 0x61, 0x63, 0x74, 0x61, 0x6c,
  0x5f, 0x6d, 0x61, 0x67, 0x6e, 0x65, 0x74, 0x28, 0x26, 0x72, 0x5f, 0x72,
  0x65, 0x61, 0x6c, 0x2c, 0x20, 0x26, 0x72, 0x5f, 0x69, 0x6d, 0x61, 0x67,
  0x2c, 0x20, 0x7a, 0x5f, 0x72, 0x65, 0x61, 0x6c, 0x2c, 0x20, 0x7a, 0x5f,
  0x69, 0x6d, 0x61, 0x67, 0x2c, 0x20, 0x66, 0x70, 0x2d, 0x3e, 0x63, 0x5f,
  0x72, 0x65, 0x61, 0x6c, 0x2c, 0x20, 0x66, 0x70, 0x2d, 0x3e, 0x63, 0x5f,
  0x69, 0x6d, 0x61, 0x67, 0x29, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
  0x62, 0x72, 0x65, 0x61, 0x6b, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x63, 0x61, 0x73, 0x65,
  0x20, 0x46, 0x43, 0x5f, 0x46, 0x52, 0x41, 0x43, 0x5f, 0x5a, 0x32, 0x5f,
  0x5a, 0x3a, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x66, 0x72, 0x61, 0x63,
  0x74, 0x61, 0x6c, 0x5f, 0x7a, 0x32, 0x5f, 0x7a, 0x28, 0x26, 0x72, 0x5f,
  0x72, 0x65, 0x61, 0x6c, 0x2c, 0x20, 0x26, 0x72, 0x5f, 0x69, 0x6d, 0x61,
  0x67, 0x2c, 0x20, 0x7a, 0x5f, 0x72, 0x65, 0x61, 0x6c, 0x2c, 0x20, 0x7a,
  0x5f, 0x69, 0x6d, 0x61, 0x67, 0x2c, 0x20, 0x66, 0x70, 0x2d, 0x3e, 0x63,
  0x5f, 0x72, 0x65, 0x61, 0x6c, 0x2c, 0x20, 0x66, 0x70, 0x2d, 0x3e, 0x63,
  0x5f, 0x69, 0x6d, 0x61, 0x67, 0x29, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x62, 0x72, 0x65, 0x61, 0x6b, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x64, 0x65, 0x66,
  0x61, 0x75, 0x6c, 0x74, 0x3a, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x62,
  0x72, 0x65, 0x61, 0x6b, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x7d, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x7a, 0x5f, 0x72, 0x65, 0x61, 0x6c, 0x20, 0x3d, 0x20, 0x72,
  0x5f, 0x72, 0x65, 0x61, 0x6c, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20,
  0x20, 0x20, 0x20, 0x20, 0x7a, 0x5f, 0x69, 0x6d, 0x61, 0x67, 0x20, 0x3d,
  0x20, 0x72, 0x5f, 0x69, 0x6d, 0x61, 0x67, 0x3b, 0x0d, 0x0a, 0x20, 0x20,
  0x20, 0x20, 0x7d, 0x0d, 0x0a, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x4d,
  0x5b, 0x79, 0x20, 0x2a, 0x20, 0x28, 0x77, 0x69, 0x64, 0x74, 0x68, 0x29,
  0x20, 0x2b, 0x20, 0x78, 0x5d, 0x20, 0x20, 0x3d, 0x20, 0x69, 0x20, 0x3d,
  0x3d, 0x20, 0x66, 0x70, 0x2d, 0x3e, 0x6d, 0x61, 0x78, 0x5f, 0x69, 0x74,
  0x65, 0x72, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x73, 0x20, 0x3f, 0x20, 0x30,
  0x20, 0x3a, 0x20, 0x69, 0x3b, 0x0d, 0x0a, 0x7d, 0x0d, 0x0a
};
unsigned int src_kernels_julia_cl_len = 2362;
