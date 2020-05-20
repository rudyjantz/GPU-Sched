; ModuleID = './VecAdd.ll'
source_filename = "/tmp/tmpxft_00000907_00000000-5_VecAdd.cudafe1.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

module asm ".section .nv_fatbin, \22a\22"
module asm ".align 8"
module asm "fatbinData:"
module asm ".quad 0x00100001ba55ed50,0x00000000000009a0,0x0000004001010002,0x0000000000000768"
module asm ".quad 0x0000000000000000,0x0000001e00010007,0x0000000000000000,0x0000000000000011"
module asm ".quad 0x0000000000000000,0x0000000000000000,0x33010102464c457f,0x0000000000000007"
module asm ".quad 0x0000006600be0002,0x0000000000000000,0x00000000000006c0,0x00000000000004c0"
module asm ".quad 0x00380040001e051e,0x0001000800400003,0x7472747368732e00,0x747274732e006261"
module asm ".quad 0x746d79732e006261,0x746d79732e006261,0x78646e68735f6261,0x666e692e766e2e00"
module asm ".quad 0x2e747865742e006f,0x6441636556365a5f,0x005f535f53665064,0x6f666e692e766e2e"
module asm ".quad 0x41636556365a5f2e,0x5f535f5366506464,0x6168732e766e2e00,0x56365a5f2e646572"
module asm ".quad 0x5366506464416365,0x2e766e2e005f535f,0x746e6174736e6f63,0x636556365a5f2e30"
module asm ".quad 0x535f536650646441,0x747368732e00005f,0x74732e0062617472,0x79732e0062617472"
module asm ".quad 0x79732e006261746d,0x6e68735f6261746d,0x692e766e2e007864,0x56365a5f006f666e"
module asm ".quad 0x5366506464416365,0x7865742e005f535f,0x636556365a5f2e74,0x535f536650646441"
module asm ".quad 0x6e692e766e2e005f,0x6556365a5f2e6f66,0x5f53665064644163,0x732e766e2e005f53"
module asm ".quad 0x5a5f2e6465726168,0x5064644163655636,0x6e2e005f535f5366,0x6174736e6f632e76"
module asm ".quad 0x56365a5f2e30746e,0x5366506464416365,0x7261705f005f535f,0x0000000000006d61"
module asm ".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0007000300000042"
module asm ".quad 0x0000000000000000,0x0000000000000000,0x000600030000008c,0x0000000000000000"
module asm ".quad 0x0000000000000000,0x0007101200000032,0x0000000000000000,0x00000000000000c0"
module asm ".quad 0x0000000300082f04,0x0008230400000008,0x0000000000000003,0x0000000300081204"
module asm ".quad 0x0008110400000000,0x0000000000000003,0x0000000200080a04,0x0018190300180140"
module asm ".quad 0x00000000000c1704,0x0021f00000100002,0x00000000000c1704,0x0021f00000080001"
module asm ".quad 0x00000000000c1704,0x0021f00000000000,0x00041d04003f1b03,0x00041c0400000010"
module asm ".quad 0x0000000000000090,0x0000000000000000,0x0000000000000000,0x0000000000000000"
module asm ".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000"
module asm ".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000"
module asm ".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000"
module asm ".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000"
module asm ".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000"
module asm ".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000"
module asm ".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000"
module asm ".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000"
module asm ".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000"
module asm ".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000"
module asm ".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000"
module asm ".quad 0x0000000000000000,0x0000000000000000,0x2282c28282304307,0x2800400110005de4"
module asm ".quad 0x2c00000094001c04,0x180000001001dde2,0x2c0000008400dc04,0x20064000a0001ca3"
module asm ".quad 0x4001400500009c43,0x208e80051000dce3,0x2283f2c04282c047,0x4001400520011c43"
module asm ".quad 0x8400000000209c85,0x208e800530015ce3,0x8400000000411c85,0x4001400540019c43"
module asm ".quad 0x208e80055001dce3,0x5000000010201c00,0x200000000002f047,0x9400000000601c85"
module asm ".quad 0x8000000000001de7,0x4003ffffe0001de7,0x4000000000001de4,0x4000000000001de4"
module asm ".quad 0x4000000000001de4,0x4000000000001de4,0x0000000000000000,0x0000000000000000"
module asm ".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000"
module asm ".quad 0x0000000000000000,0x0000000000000000,0x0000000300000001,0x0000000000000000"
module asm ".quad 0x0000000000000000,0x0000000000000040,0x000000000000009a,0x0000000000000000"
module asm ".quad 0x0000000000000001,0x0000000000000000,0x000000030000000b,0x0000000000000000"
module asm ".quad 0x0000000000000000,0x00000000000000da,0x00000000000000b1,0x0000000000000000"
module asm ".quad 0x0000000000000001,0x0000000000000000,0x0000000200000013,0x0000000000000000"
module asm ".quad 0x0000000000000000,0x0000000000000190,0x0000000000000060,0x0000000200000002"
module asm ".quad 0x0000000000000008,0x0000000000000018,0x7000000000000029,0x0000000000000000"
module asm ".quad 0x0000000000000000,0x00000000000001f0,0x0000000000000030,0x0000000000000003"
module asm ".quad 0x0000000000000004,0x0000000000000000,0x7000000000000048,0x0000000000000000"
module asm ".quad 0x0000000000000000,0x0000000000000220,0x0000000000000054,0x0000000700000003"
module asm ".quad 0x0000000000000004,0x0000000000000000,0x000000010000007c,0x0000000000000002"
module asm ".quad 0x0000000000000000,0x0000000000000274,0x0000000000000158,0x0000000700000000"
module asm ".quad 0x0000000000000004,0x0000000000000000,0x0000000100000032,0x0000000000000006"
module asm ".quad 0x0000000000000000,0x0000000000000400,0x00000000000000c0,0x0800000300000003"
module asm ".quad 0x0000000000000040,0x0000000000000000,0x0000000500000006,0x00000000000006c0"
module asm ".quad 0x0000000000000000,0x0000000000000000,0x00000000000000a8,0x00000000000000a8"
module asm ".quad 0x0000000000000008,0x0000000500000001,0x0000000000000274,0x0000000000000000"
module asm ".quad 0x0000000000000000,0x0000000000000218,0x0000000000000218,0x0000000000000008"
module asm ".quad 0x0000000600000001,0x0000000000000000,0x0000000000000000,0x0000000000000000"
module asm ".quad 0x0000000000000000,0x0000000000000000,0x0000000000000008,0x0000004801010001"
module asm ".quad 0x00000000000001b0,0x00000040000001ae,0x0000001e00060005,0x0000000000000000"
module asm ".quad 0x0000000000002011,0x0000000000000000,0x0000000000000326,0x0000000000000000"
module asm ".quad 0x762e1cf200010a13,0x36206e6f69737265,0x677261742e0a352e,0x30335f6d73207465"
module asm ".quad 0x7365726464612e0a,0x3620657a69735f73,0x6973691bfc002f34,0x746e652e20656c62"
module asm ".quad 0x6556365a5f207972,0x5f53665064644163,0x7261702e0a285f53,0x1d3436752e206d61"
module asm ".quad 0x305f3f001b5f1100,0x0025311f1000252c,0x7b0a290a3207f311,0x662e206765722e0a"
module asm ".quad 0x3e343c6625203233,0x360011621000113b,0x3601f20011353c72,0x31313c6472252034"
module asm ".quad 0x61646c0a0a0a3b3e,0x314f0018752e2200,0x303d0300675b202c,0x002e321f002e3b5d"
module asm ".quad 0x331f00002e311f06,0x3b5d3203f406002e,0x6f742e617476630a,0x346c61626f6c672e"
module asm ".quad 0x1f0f003a2c342100,0x321f001f35110500,0x81001f361105001f,0x752e766f6d0a3b31"
module asm ".quad 0x6325202c31b8010b,0x0017782e64696174,0x16746e25202c326c,0x001525202c334400"
module asm ".quad 0x732e6f6c2e646171,0x0100332c34230018,0x6d0a3b3372c3004f,0x21656469772e6c75"
module asm ".quad 0x8200272c37643200,0x732e6464610a3b34,0x1100962c38260090,0xb80200b003012137"
module asm ".quad 0x19002600017d0001,0x00ea2c392600355d,0x003532120000350f,0x2200150200353913"
module asm ".quad 0x32662539004f2c33,0x01562c303136004c,0x004d74730a3b3758,0x00375d1000225b11"
module asm ".quad 0x3b7465720a3b33d0, 0x0000000a0a0a7d0a"
module asm ".text"

%struct.__fatBinC_Wrapper_t = type { i32, i32, i64*, i8* }
%struct.dim3 = type { i32, i32, i32 }
%struct.CUstream_st = type opaque
%struct.uint3 = type { i32, i32, i32 }

$_ZN4dim3C2Ejjj = comdat any

$_ZSt3sinIiEN9__gnu_cxx11__enable_ifIXsr12__is_integerIT_EE7__valueEdE6__typeES2_ = comdat any

$_ZSt3cosIiEN9__gnu_cxx11__enable_ifIXsr12__is_integerIT_EE7__valueEdE6__typeES2_ = comdat any

@_ZZ29__device_stub__Z6VecAddPfS_S_PfS_S_E3__f = internal global i8* null, align 8
@_ZL15__fatDeviceText = internal constant %struct.__fatBinC_Wrapper_t { i32 1180844977, i32 1, i64* getelementptr inbounds ([310 x i64], [310 x i64]* @fatbinData, i32 0, i32 0), i8* null }, section ".nvFatBinSegment", align 8
@_ZL20__cudaFatCubinHandle = internal global i8** null, align 8
@fatbinData = external dso_local constant [310 x i64], align 16
@_ZZL31__nv_cudaEntityRegisterCallbackPPvE5__ref = internal global i8** null, align 8
@.str = private unnamed_addr constant [16 x i8] c"_Z6VecAddPfS_S_\00", align 1
@_ZL32__nv_fatbinhandle_for_managed_rt = internal global i8** null, align 8
@_ZZL22____nv_dummy_param_refPvE5__ref = internal global i8** null, align 8
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_ZL24__sti____cudaRegisterAllv, i8* null }]

; Function Attrs: noinline norecurse optnone uwtable
define dso_local i32 @main(i32 %argc, i8** %argv) #0 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  %A = alloca [512 x float], align 16
  %B = alloca [512 x float], align 16
  %C = alloca [512 x float], align 16
  %ThreadsPerBlock = alloca %struct.dim3, align 4
  %BlocksPerGrid = alloca %struct.dim3, align 4
  %size = alloca i64, align 8
  %i = alloca i32, align 4
  %d_A = alloca float*, align 8
  %d_B = alloca float*, align 8
  %d_C = alloca float*, align 8
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp14 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp14.coerce = alloca { i64, i32 }, align 4
  %agg.tmp27 = alloca %struct.dim3, align 4
  %agg.tmp28 = alloca %struct.dim3, align 4
  %agg.tmp27.coerce = alloca { i64, i32 }, align 4
  %agg.tmp28.coerce = alloca { i64, i32 }, align 4
  store i32 0, i32* %retval, align 4
  store i32 %argc, i32* %argc.addr, align 4
  store i8** %argv, i8*** %argv.addr, align 8
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %ThreadsPerBlock, i32 32, i32 1, i32 1)
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %BlocksPerGrid, i32 16, i32 1, i32 1)
  store i64 2048, i64* %size, align 8
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 512
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, i32* %i, align 4
  %call = call double @_ZSt3sinIiEN9__gnu_cxx11__enable_ifIXsr12__is_integerIT_EE7__valueEdE6__typeES2_(i32 %1)
  %2 = load i32, i32* %i, align 4
  %call1 = call double @_ZSt3sinIiEN9__gnu_cxx11__enable_ifIXsr12__is_integerIT_EE7__valueEdE6__typeES2_(i32 %2)
  %mul = fmul double %call, %call1
  %conv = fptrunc double %mul to float
  %3 = load i32, i32* %i, align 4
  %idxprom = sext i32 %3 to i64
  %arrayidx = getelementptr inbounds [512 x float], [512 x float]* %A, i64 0, i64 %idxprom
  store float %conv, float* %arrayidx, align 4
  %4 = load i32, i32* %i, align 4
  %call2 = call double @_ZSt3cosIiEN9__gnu_cxx11__enable_ifIXsr12__is_integerIT_EE7__valueEdE6__typeES2_(i32 %4)
  %5 = load i32, i32* %i, align 4
  %call3 = call double @_ZSt3cosIiEN9__gnu_cxx11__enable_ifIXsr12__is_integerIT_EE7__valueEdE6__typeES2_(i32 %5)
  %mul4 = fmul double %call2, %call3
  %conv5 = fptrunc double %mul4 to float
  %6 = load i32, i32* %i, align 4
  %idxprom6 = sext i32 %6 to i64
  %arrayidx7 = getelementptr inbounds [512 x float], [512 x float]* %B, i64 0, i64 %idxprom6
  store float %conv5, float* %arrayidx7, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %7 = load i32, i32* %i, align 4
  %inc = add nsw i32 %7, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %8 = load i64, i64* %size, align 8
  %call8 = call i32 @_ZL10cudaMallocIfE9cudaErrorPPT_m(float** %d_A, i64 %8)
  %9 = load i64, i64* %size, align 8
  %call9 = call i32 @_ZL10cudaMallocIfE9cudaErrorPPT_m(float** %d_B, i64 %9)
  %10 = load i64, i64* %size, align 8
  %call10 = call i32 @_ZL10cudaMallocIfE9cudaErrorPPT_m(float** %d_C, i64 %10)
  %11 = load float*, float** %d_A, align 8
  %12 = bitcast float* %11 to i8*
  %arraydecay = getelementptr inbounds [512 x float], [512 x float]* %A, i32 0, i32 0
  %13 = bitcast float* %arraydecay to i8*
  %14 = load i64, i64* %size, align 8
  %call11 = call i32 @cudaMemcpy(i8* %12, i8* %13, i64 %14, i32 1)
  %15 = load float*, float** %d_B, align 8
  %16 = bitcast float* %15 to i8*
  %arraydecay12 = getelementptr inbounds [512 x float], [512 x float]* %B, i32 0, i32 0
  %17 = bitcast float* %arraydecay12 to i8*
  %18 = load i64, i64* %size, align 8
  %call13 = call i32 @cudaMemcpy(i8* %16, i8* %17, i64 %18, i32 1)
  %19 = bitcast %struct.dim3* %agg.tmp to i8*
  %20 = bitcast %struct.dim3* %BlocksPerGrid to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %19, i8* align 4 %20, i64 12, i1 false)
  %21 = bitcast %struct.dim3* %agg.tmp14 to i8*
  %22 = bitcast %struct.dim3* %ThreadsPerBlock to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %21, i8* align 4 %22, i64 12, i1 false)
  %23 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*
  %24 = bitcast %struct.dim3* %agg.tmp to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %23, i8* align 4 %24, i64 12, i1 false)
  %25 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0
  %26 = load i64, i64* %25, align 4
  %27 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1
  %28 = load i32, i32* %27, align 4
  %29 = bitcast { i64, i32 }* %agg.tmp14.coerce to i8*
  %30 = bitcast %struct.dim3* %agg.tmp14 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %29, i8* align 4 %30, i64 12, i1 false)
  %31 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp14.coerce, i32 0, i32 0
  %32 = load i64, i64* %31, align 4
  %33 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp14.coerce, i32 0, i32 1
  %34 = load i32, i32* %33, align 4
  %call15 = call i32 @__cudaPushCallConfiguration(i64 %26, i32 %28, i64 %32, i32 %34, i64 0, %struct.CUstream_st* null)
  %tobool = icmp ne i32 %call15, 0
  br i1 %tobool, label %cond.true, label %cond.false

cond.true:                                        ; preds = %for.end
  br label %cond.end

cond.false:                                       ; preds = %for.end
  %35 = load float*, float** %d_A, align 8
  %36 = load float*, float** %d_B, align 8
  %37 = load float*, float** %d_C, align 8
  call void @_Z6VecAddPfS_S_(float* %35, float* %36, float* %37)
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %arraydecay16 = getelementptr inbounds [512 x float], [512 x float]* %C, i32 0, i32 0
  %38 = bitcast float* %arraydecay16 to i8*
  %39 = load float*, float** %d_C, align 8
  %40 = bitcast float* %39 to i8*
  %41 = load i64, i64* %size, align 8
  %call17 = call i32 @cudaMemcpy(i8* %38, i8* %40, i64 %41, i32 2)
  %42 = load float*, float** %d_A, align 8
  %43 = bitcast float* %42 to i8*
  %call18 = call i32 @cudaFree(i8* %43)
  %44 = load float*, float** %d_B, align 8
  %45 = bitcast float* %44 to i8*
  %call19 = call i32 @cudaFree(i8* %45)
  %46 = load float*, float** %d_C, align 8
  %47 = bitcast float* %46 to i8*
  %call20 = call i32 @cudaFree(i8* %47)
  %48 = load i64, i64* %size, align 8
  %mul21 = mul i64 %48, 2
  %call22 = call i32 @_ZL10cudaMallocIfE9cudaErrorPPT_m(float** %d_A, i64 %mul21)
  %49 = load i64, i64* %size, align 8
  %mul23 = mul i64 %49, 2
  %call24 = call i32 @_ZL10cudaMallocIfE9cudaErrorPPT_m(float** %d_B, i64 %mul23)
  %50 = load i64, i64* %size, align 8
  %mul25 = mul i64 %50, 2
  %call26 = call i32 @_ZL10cudaMallocIfE9cudaErrorPPT_m(float** %d_C, i64 %mul25)
  %51 = bitcast %struct.dim3* %agg.tmp27 to i8*
  %52 = bitcast %struct.dim3* %BlocksPerGrid to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %51, i8* align 4 %52, i64 12, i1 false)
  %53 = bitcast %struct.dim3* %agg.tmp28 to i8*
  %54 = bitcast %struct.dim3* %ThreadsPerBlock to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %53, i8* align 4 %54, i64 12, i1 false)
  %55 = bitcast { i64, i32 }* %agg.tmp27.coerce to i8*
  %56 = bitcast %struct.dim3* %agg.tmp27 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %55, i8* align 4 %56, i64 12, i1 false)
  %57 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp27.coerce, i32 0, i32 0
  %58 = load i64, i64* %57, align 4
  %59 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp27.coerce, i32 0, i32 1
  %60 = load i32, i32* %59, align 4
  %61 = bitcast { i64, i32 }* %agg.tmp28.coerce to i8*
  %62 = bitcast %struct.dim3* %agg.tmp28 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %61, i8* align 4 %62, i64 12, i1 false)
  %63 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp28.coerce, i32 0, i32 0
  %64 = load i64, i64* %63, align 4
  %65 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp28.coerce, i32 0, i32 1
  %66 = load i32, i32* %65, align 4
  %call29 = call i32 @__cudaPushCallConfiguration(i64 %58, i32 %60, i64 %64, i32 %66, i64 0, %struct.CUstream_st* null)
  %tobool30 = icmp ne i32 %call29, 0
  br i1 %tobool30, label %cond.true31, label %cond.false32

cond.true31:                                      ; preds = %cond.end
  br label %cond.end33

cond.false32:                                     ; preds = %cond.end
  %67 = load float*, float** %d_A, align 8
  %68 = load float*, float** %d_B, align 8
  %69 = load float*, float** %d_C, align 8
  call void @_Z6VecAddPfS_S_(float* %67, float* %68, float* %69)
  br label %cond.end33

cond.end33:                                       ; preds = %cond.false32, %cond.true31
  %70 = load float*, float** %d_A, align 8
  %71 = bitcast float* %70 to i8*
  %call34 = call i32 @cudaFree(i8* %71)
  %72 = load float*, float** %d_B, align 8
  %73 = bitcast float* %72 to i8*
  %call35 = call i32 @cudaFree(i8* %73)
  %74 = load float*, float** %d_C, align 8
  %75 = bitcast float* %74 to i8*
  %call36 = call i32 @cudaFree(i8* %75)
  ret i32 0
}

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN4dim3C2Ejjj(%struct.dim3* %this, i32 %vx, i32 %vy, i32 %vz) unnamed_addr #1 comdat align 2 {
entry:
  %this.addr = alloca %struct.dim3*, align 8
  %vx.addr = alloca i32, align 4
  %vy.addr = alloca i32, align 4
  %vz.addr = alloca i32, align 4
  store %struct.dim3* %this, %struct.dim3** %this.addr, align 8
  store i32 %vx, i32* %vx.addr, align 4
  store i32 %vy, i32* %vy.addr, align 4
  store i32 %vz, i32* %vz.addr, align 4
  %this1 = load %struct.dim3*, %struct.dim3** %this.addr, align 8
  %x = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 0
  %0 = load i32, i32* %vx.addr, align 4
  store i32 %0, i32* %x, align 4
  %y = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 1
  %1 = load i32, i32* %vy.addr, align 4
  store i32 %1, i32* %y, align 4
  %z = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 2
  %2 = load i32, i32* %vz.addr, align 4
  store i32 %2, i32* %z, align 4
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local double @_ZSt3sinIiEN9__gnu_cxx11__enable_ifIXsr12__is_integerIT_EE7__valueEdE6__typeES2_(i32 %__x) #1 comdat {
entry:
  %__x.addr = alloca i32, align 4
  store i32 %__x, i32* %__x.addr, align 4
  %0 = load i32, i32* %__x.addr, align 4
  %conv = sitofp i32 %0 to double
  %call = call double @sin(double %conv) #6
  ret double %call
}

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local double @_ZSt3cosIiEN9__gnu_cxx11__enable_ifIXsr12__is_integerIT_EE7__valueEdE6__typeES2_(i32 %__x) #1 comdat {
entry:
  %__x.addr = alloca i32, align 4
  store i32 %__x, i32* %__x.addr, align 4
  %0 = load i32, i32* %__x.addr, align 4
  %conv = sitofp i32 %0 to double
  %call = call double @cos(double %conv) #6
  ret double %call
}

; Function Attrs: noinline optnone uwtable
define internal i32 @_ZL10cudaMallocIfE9cudaErrorPPT_m(float** %devPtr, i64 %size) #2 {
entry:
  %devPtr.addr = alloca float**, align 8
  %size.addr = alloca i64, align 8
  store float** %devPtr, float*** %devPtr.addr, align 8
  store i64 %size, i64* %size.addr, align 8
  %0 = load float**, float*** %devPtr.addr, align 8
  %1 = bitcast float** %0 to i8*
  %2 = bitcast i8* %1 to i8**
  %3 = load i64, i64* %size.addr, align 8
  %call = call i32 @cudaMalloc(i8** %2, i64 %3)
  ret i32 %call
}

declare dso_local i32 @cudaMemcpy(i8*, i8*, i64, i32) #3

declare dso_local i32 @__cudaPushCallConfiguration(i64, i32, i64, i32, i64, %struct.CUstream_st*) #3

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1 immarg) #4

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z6VecAddPfS_S_(float* %__cuda_0, float* %__cuda_1, float* %__cuda_2) #2 {
entry:
  %__cuda_0.addr = alloca float*, align 8
  %__cuda_1.addr = alloca float*, align 8
  %__cuda_2.addr = alloca float*, align 8
  store float* %__cuda_0, float** %__cuda_0.addr, align 8
  store float* %__cuda_1, float** %__cuda_1.addr, align 8
  store float* %__cuda_2, float** %__cuda_2.addr, align 8
  %0 = load float*, float** %__cuda_0.addr, align 8
  %1 = load float*, float** %__cuda_1.addr, align 8
  %2 = load float*, float** %__cuda_2.addr, align 8
  call void @_Z29__device_stub__Z6VecAddPfS_S_PfS_S_(float* %0, float* %1, float* %2)
  ret void
}

declare dso_local i32 @cudaFree(i8*) #3

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z29__device_stub__Z6VecAddPfS_S_PfS_S_(float* %__par0, float* %__par1, float* %__par2) #2 {
entry:
  %__par0.addr = alloca float*, align 8
  %__par1.addr = alloca float*, align 8
  %__par2.addr = alloca float*, align 8
  %__args_arr = alloca [3 x i8*], align 16
  %__args_idx = alloca i32, align 4
  %__gridDim = alloca %struct.dim3, align 4
  %__blockDim = alloca %struct.dim3, align 4
  %__sharedMem = alloca i64, align 8
  %__stream = alloca %struct.CUstream_st*, align 8
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp9 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp9.coerce = alloca { i64, i32 }, align 4
  %agg.tmp13 = alloca %struct.dim3, align 4
  %agg.tmp14 = alloca %struct.dim3, align 4
  %agg.tmp13.coerce = alloca { i64, i32 }, align 4
  %agg.tmp14.coerce = alloca { i64, i32 }, align 4
  store float* %__par0, float** %__par0.addr, align 8
  store float* %__par1, float** %__par1.addr, align 8
  store float* %__par2, float** %__par2.addr, align 8
  store i32 0, i32* %__args_idx, align 4
  %0 = bitcast float** %__par0.addr to i8*
  %1 = load i32, i32* %__args_idx, align 4
  %idxprom = sext i32 %1 to i64
  %arrayidx = getelementptr inbounds [3 x i8*], [3 x i8*]* %__args_arr, i64 0, i64 %idxprom
  store i8* %0, i8** %arrayidx, align 8
  %2 = load i32, i32* %__args_idx, align 4
  %inc = add nsw i32 %2, 1
  store i32 %inc, i32* %__args_idx, align 4
  %3 = bitcast float** %__par1.addr to i8*
  %4 = load i32, i32* %__args_idx, align 4
  %idxprom1 = sext i32 %4 to i64
  %arrayidx2 = getelementptr inbounds [3 x i8*], [3 x i8*]* %__args_arr, i64 0, i64 %idxprom1
  store i8* %3, i8** %arrayidx2, align 8
  %5 = load i32, i32* %__args_idx, align 4
  %inc3 = add nsw i32 %5, 1
  store i32 %inc3, i32* %__args_idx, align 4
  %6 = bitcast float** %__par2.addr to i8*
  %7 = load i32, i32* %__args_idx, align 4
  %idxprom4 = sext i32 %7 to i64
  %arrayidx5 = getelementptr inbounds [3 x i8*], [3 x i8*]* %__args_arr, i64 0, i64 %idxprom4
  store i8* %6, i8** %arrayidx5, align 8
  %8 = load i32, i32* %__args_idx, align 4
  %inc6 = add nsw i32 %8, 1
  store i32 %inc6, i32* %__args_idx, align 4
  store i8* bitcast (void (float*, float*, float*)* @_Z6VecAddPfS_S_ to i8*), i8** @_ZZ29__device_stub__Z6VecAddPfS_S_PfS_S_E3__f, align 8
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %__gridDim, i32 1, i32 1, i32 1)
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %__blockDim, i32 1, i32 1, i32 1)
  %9 = bitcast %struct.CUstream_st** %__stream to i8*
  %call = call i32 @__cudaPopCallConfiguration(%struct.dim3* %__gridDim, %struct.dim3* %__blockDim, i64* %__sharedMem, i8* %9)
  %cmp = icmp ne i32 %call, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  br label %if.end17

if.end:                                           ; preds = %entry
  %10 = load i32, i32* %__args_idx, align 4
  %cmp7 = icmp eq i32 %10, 0
  br i1 %cmp7, label %if.then8, label %if.else

if.then8:                                         ; preds = %if.end
  %11 = bitcast %struct.dim3* %agg.tmp to i8*
  %12 = bitcast %struct.dim3* %__gridDim to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %11, i8* align 4 %12, i64 12, i1 false)
  %13 = bitcast %struct.dim3* %agg.tmp9 to i8*
  %14 = bitcast %struct.dim3* %__blockDim to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %13, i8* align 4 %14, i64 12, i1 false)
  %15 = load i32, i32* %__args_idx, align 4
  %idxprom10 = sext i32 %15 to i64
  %arrayidx11 = getelementptr inbounds [3 x i8*], [3 x i8*]* %__args_arr, i64 0, i64 %idxprom10
  %16 = load i64, i64* %__sharedMem, align 8
  %17 = load %struct.CUstream_st*, %struct.CUstream_st** %__stream, align 8
  %18 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*
  %19 = bitcast %struct.dim3* %agg.tmp to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %18, i8* align 4 %19, i64 12, i1 false)
  %20 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0
  %21 = load i64, i64* %20, align 4
  %22 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1
  %23 = load i32, i32* %22, align 4
  %24 = bitcast { i64, i32 }* %agg.tmp9.coerce to i8*
  %25 = bitcast %struct.dim3* %agg.tmp9 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %24, i8* align 4 %25, i64 12, i1 false)
  %26 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp9.coerce, i32 0, i32 0
  %27 = load i64, i64* %26, align 4
  %28 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp9.coerce, i32 0, i32 1
  %29 = load i32, i32* %28, align 4
  %call12 = call i32 @_ZL16cudaLaunchKernelIcE9cudaErrorPKT_4dim3S4_PPvmP11CUstream_st(i8* bitcast (void (float*, float*, float*)* @_Z6VecAddPfS_S_ to i8*), i64 %21, i32 %23, i64 %27, i32 %29, i8** %arrayidx11, i64 %16, %struct.CUstream_st* %17)
  br label %if.end17

if.else:                                          ; preds = %if.end
  %30 = bitcast %struct.dim3* %agg.tmp13 to i8*
  %31 = bitcast %struct.dim3* %__gridDim to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %30, i8* align 4 %31, i64 12, i1 false)
  %32 = bitcast %struct.dim3* %agg.tmp14 to i8*
  %33 = bitcast %struct.dim3* %__blockDim to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %32, i8* align 4 %33, i64 12, i1 false)
  %arrayidx15 = getelementptr inbounds [3 x i8*], [3 x i8*]* %__args_arr, i64 0, i64 0
  %34 = load i64, i64* %__sharedMem, align 8
  %35 = load %struct.CUstream_st*, %struct.CUstream_st** %__stream, align 8
  %36 = bitcast { i64, i32 }* %agg.tmp13.coerce to i8*
  %37 = bitcast %struct.dim3* %agg.tmp13 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %36, i8* align 4 %37, i64 12, i1 false)
  %38 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp13.coerce, i32 0, i32 0
  %39 = load i64, i64* %38, align 4
  %40 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp13.coerce, i32 0, i32 1
  %41 = load i32, i32* %40, align 4
  %42 = bitcast { i64, i32 }* %agg.tmp14.coerce to i8*
  %43 = bitcast %struct.dim3* %agg.tmp14 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %42, i8* align 4 %43, i64 12, i1 false)
  %44 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp14.coerce, i32 0, i32 0
  %45 = load i64, i64* %44, align 4
  %46 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp14.coerce, i32 0, i32 1
  %47 = load i32, i32* %46, align 4
  %call16 = call i32 @_ZL16cudaLaunchKernelIcE9cudaErrorPKT_4dim3S4_PPvmP11CUstream_st(i8* bitcast (void (float*, float*, float*)* @_Z6VecAddPfS_S_ to i8*), i64 %39, i32 %41, i64 %45, i32 %47, i8** %arrayidx15, i64 %34, %struct.CUstream_st* %35)
  br label %if.end17

if.end17:                                         ; preds = %if.else, %if.then8, %if.then
  ret void
}

declare dso_local i32 @__cudaPopCallConfiguration(%struct.dim3*, %struct.dim3*, i64*, i8*) #3

; Function Attrs: noinline optnone uwtable
define internal i32 @_ZL16cudaLaunchKernelIcE9cudaErrorPKT_4dim3S4_PPvmP11CUstream_st(i8* %func, i64 %gridDim.coerce0, i32 %gridDim.coerce1, i64 %blockDim.coerce0, i32 %blockDim.coerce1, i8** %args, i64 %sharedMem, %struct.CUstream_st* %stream) #2 {
entry:
  %gridDim = alloca %struct.dim3, align 4
  %coerce = alloca { i64, i32 }, align 4
  %blockDim = alloca %struct.dim3, align 4
  %coerce1 = alloca { i64, i32 }, align 4
  %func.addr = alloca i8*, align 8
  %args.addr = alloca i8**, align 8
  %sharedMem.addr = alloca i64, align 8
  %stream.addr = alloca %struct.CUstream_st*, align 8
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp2 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp2.coerce = alloca { i64, i32 }, align 4
  %0 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %coerce, i32 0, i32 0
  store i64 %gridDim.coerce0, i64* %0, align 4
  %1 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %coerce, i32 0, i32 1
  store i32 %gridDim.coerce1, i32* %1, align 4
  %2 = bitcast %struct.dim3* %gridDim to i8*
  %3 = bitcast { i64, i32 }* %coerce to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false)
  %4 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %coerce1, i32 0, i32 0
  store i64 %blockDim.coerce0, i64* %4, align 4
  %5 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %coerce1, i32 0, i32 1
  store i32 %blockDim.coerce1, i32* %5, align 4
  %6 = bitcast %struct.dim3* %blockDim to i8*
  %7 = bitcast { i64, i32 }* %coerce1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %6, i8* align 4 %7, i64 12, i1 false)
  store i8* %func, i8** %func.addr, align 8
  store i8** %args, i8*** %args.addr, align 8
  store i64 %sharedMem, i64* %sharedMem.addr, align 8
  store %struct.CUstream_st* %stream, %struct.CUstream_st** %stream.addr, align 8
  %8 = load i8*, i8** %func.addr, align 8
  %9 = bitcast %struct.dim3* %agg.tmp to i8*
  %10 = bitcast %struct.dim3* %gridDim to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %9, i8* align 4 %10, i64 12, i1 false)
  %11 = bitcast %struct.dim3* %agg.tmp2 to i8*
  %12 = bitcast %struct.dim3* %blockDim to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %11, i8* align 4 %12, i64 12, i1 false)
  %13 = load i8**, i8*** %args.addr, align 8
  %14 = load i64, i64* %sharedMem.addr, align 8
  %15 = load %struct.CUstream_st*, %struct.CUstream_st** %stream.addr, align 8
  %16 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*
  %17 = bitcast %struct.dim3* %agg.tmp to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %16, i8* align 4 %17, i64 12, i1 false)
  %18 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0
  %19 = load i64, i64* %18, align 4
  %20 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1
  %21 = load i32, i32* %20, align 4
  %22 = bitcast { i64, i32 }* %agg.tmp2.coerce to i8*
  %23 = bitcast %struct.dim3* %agg.tmp2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %22, i8* align 4 %23, i64 12, i1 false)
  %24 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp2.coerce, i32 0, i32 0
  %25 = load i64, i64* %24, align 4
  %26 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp2.coerce, i32 0, i32 1
  %27 = load i32, i32* %26, align 4
  %call = call i32 @cudaLaunchKernel(i8* %8, i64 %19, i32 %21, i64 %25, i32 %27, i8** %13, i64 %14, %struct.CUstream_st* %15)
  ret i32 %call
}

; Function Attrs: noinline optnone uwtable
define internal void @_ZL24__sti____cudaRegisterAllv() #2 {
entry:
  %callback_fp = alloca void (i8**)*, align 8
  %call = call i8** @__cudaRegisterFatBinary(i8* bitcast (%struct.__fatBinC_Wrapper_t* @_ZL15__fatDeviceText to i8*))
  store i8** %call, i8*** @_ZL20__cudaFatCubinHandle, align 8
  store void (i8**)* @_ZL31__nv_cudaEntityRegisterCallbackPPv, void (i8**)** %callback_fp, align 8
  %0 = load void (i8**)*, void (i8**)** %callback_fp, align 8
  %1 = load i8**, i8*** @_ZL20__cudaFatCubinHandle, align 8
  call void %0(i8** %1)
  %2 = load i8**, i8*** @_ZL20__cudaFatCubinHandle, align 8
  call void @__cudaRegisterFatBinaryEnd(i8** %2)
  %call1 = call i32 @atexit(void ()* @_ZL26__cudaUnregisterBinaryUtilv) #6
  ret void
}

declare dso_local i8** @__cudaRegisterFatBinary(i8*) #3

; Function Attrs: noinline optnone uwtable
define internal void @_ZL31__nv_cudaEntityRegisterCallbackPPv(i8** %__T0) #2 {
entry:
  %__T0.addr = alloca i8**, align 8
  store i8** %__T0, i8*** %__T0.addr, align 8
  %0 = load i8**, i8*** %__T0.addr, align 8
  store i8** %0, i8*** @_ZZL31__nv_cudaEntityRegisterCallbackPPvE5__ref, align 8
  %1 = load i8**, i8*** %__T0.addr, align 8
  call void @_ZL37__nv_save_fatbinhandle_for_managed_rtPPv(i8** %1)
  %2 = load i8**, i8*** %__T0.addr, align 8
  call void @__cudaRegisterFunction(i8** %2, i8* bitcast (void (float*, float*, float*)* @_Z6VecAddPfS_S_ to i8*), i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str, i32 0, i32 0), i32 -1, %struct.uint3* null, %struct.uint3* null, %struct.dim3* null, %struct.dim3* null, i32* null)
  ret void
}

declare dso_local void @__cudaRegisterFatBinaryEnd(i8**) #3

; Function Attrs: nounwind
declare dso_local i32 @atexit(void ()*) #5

; Function Attrs: noinline optnone uwtable
define internal void @_ZL26__cudaUnregisterBinaryUtilv() #2 {
entry:
  call void @_ZL22____nv_dummy_param_refPv(i8* bitcast (i8*** @_ZL20__cudaFatCubinHandle to i8*))
  %0 = load i8**, i8*** @_ZL20__cudaFatCubinHandle, align 8
  call void @__cudaUnregisterFatBinary(i8** %0)
  ret void
}

; Function Attrs: nounwind
declare dso_local double @sin(double) #5

; Function Attrs: nounwind
declare dso_local double @cos(double) #5

; Function Attrs: noinline nounwind optnone uwtable
define internal void @_ZL37__nv_save_fatbinhandle_for_managed_rtPPv(i8** %in) #1 {
entry:
  %in.addr = alloca i8**, align 8
  store i8** %in, i8*** %in.addr, align 8
  %0 = load i8**, i8*** %in.addr, align 8
  store i8** %0, i8*** @_ZL32__nv_fatbinhandle_for_managed_rt, align 8
  ret void
}

declare dso_local void @__cudaRegisterFunction(i8**, i8*, i8*, i8*, i32, %struct.uint3*, %struct.uint3*, %struct.dim3*, %struct.dim3*, i32*) #3

; Function Attrs: noinline nounwind optnone uwtable
define internal void @_ZL22____nv_dummy_param_refPv(i8* %param) #1 {
entry:
  %param.addr = alloca i8*, align 8
  store i8* %param, i8** %param.addr, align 8
  %0 = load i8*, i8** %param.addr, align 8
  %1 = bitcast i8* %0 to i8**
  store i8** %1, i8*** @_ZZL22____nv_dummy_param_refPvE5__ref, align 8
  ret void
}

declare dso_local void @__cudaUnregisterFatBinary(i8**) #3

declare dso_local i32 @cudaMalloc(i8**, i64) #3

declare dso_local i32 @cudaLaunchKernel(i8*, i64, i32, i64, i32, i8**, i64, %struct.CUstream_st*) #3

attributes #0 = { noinline norecurse optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { noinline optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { argmemonly nounwind }
attributes #5 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 8.0.0 (tags/RELEASE_800/final)"}
