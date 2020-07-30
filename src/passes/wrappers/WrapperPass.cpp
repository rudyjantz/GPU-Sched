
#include "WrapperPass.h"

#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>

using namespace llvm;

void WrapperPass::replaceMalloc(CallInst *CI) {
  dbgs() << "repalcing Malloc : " << *CI << "\n\t" << *CI->getArgOperand(0)
         << "\n\t" << *CI->getArgOperand(1) << "\n\n";
  IRBuilder<NoFolder> IRB(CI->getContext());
  IRB.SetInsertPoint(CI);
  SmallVector<Value *, 2> args;
  args.push_back(CI->getArgOperand(0));
  args.push_back(CI->getArgOperand(1));
  auto ret = IRB.CreateCall(MallocWrapper, args);
  CI->replaceAllUsesWith(ret);
}

void WrapperPass::replaceMemcpy(CallInst *CI) {
  dbgs() << "repalcing Memcpy : " << *CI << "\n\t" << *CI->getArgOperand(0)
         << "\n\t" << *CI->getArgOperand(1) << "\n\t" << *CI->getArgOperand(2)
         << "\n\t" << *CI->getArgOperand(3) << "\n\n";
  IRBuilder<NoFolder> IRB(CI->getContext());
  IRB.SetInsertPoint(CI);
  SmallVector<Value *, 4> args;
  args.push_back(CI->getArgOperand(0));
  args.push_back(CI->getArgOperand(1));
  args.push_back(CI->getArgOperand(2));
  args.push_back(CI->getArgOperand(3));
  auto ret = IRB.CreateCall(MemcpyWrapper, args);
  CI->replaceAllUsesWith(ret);
}

void WrapperPass::replaceFree(CallInst *CI) {
  dbgs() << "repalcing Free : " << *CI << "\n\t" << *CI->getArgOperand(0)
         << "\n\n";
  IRBuilder<NoFolder> IRB(CI->getContext());
  IRB.SetInsertPoint(CI);
  SmallVector<Value *, 2> args;
  args.push_back(CI->getArgOperand(0));
  auto ret = IRB.CreateCall(FreeWrapper, args);
  CI->replaceAllUsesWith(ret);
}

void WrapperPass::addKernelLaunchPrepare(CallInst *CI) {
  dbgs() << "adding LaunchPrepare : " << *CI << "\n\t" << *CI->getArgOperand(0)
         << "\n\t" << *CI->getArgOperand(1) << "\n\t" << *CI->getArgOperand(2)
         << "\n\t" << *CI->getArgOperand(3) << "\n\n";
  IRBuilder<NoFolder> IRB(CI->getContext());
  IRB.SetInsertPoint(CI);
  SmallVector<Value *, 4> args;
  args.push_back(CI->getArgOperand(0));
  args.push_back(CI->getArgOperand(1));
  args.push_back(CI->getArgOperand(2));
  args.push_back(CI->getArgOperand(3));
  auto ret = IRB.CreateCall(KernelLaunchPrepare, args);
  // CI->replaceAllUsesWith(ret);
}

bool WrapperPass::doInitialization(Module &M) {
  auto &ctx = M.getContext();

  Type *sizeTy = Type::getInt64Ty(ctx);
  Type *retTy = Type::getInt32Ty(ctx);

  // declare cudaMallocWrapper
  Type *ptrTy = Type::getInt8PtrTy(ctx)->getPointerTo();
  FunctionType *MallocWrapperFTy =
      FunctionType::get(retTy, {ptrTy, sizeTy}, false);
  MallocWrapper = M.getOrInsertFunction("cudaMallocWrapper", MallocWrapperFTy);

  // declare cudaMemcpyWrapper
  Type *dstTy = Type::getInt8PtrTy(ctx);
  Type *srcTy = Type::getInt8PtrTy(ctx);
  Type *kindTy = Type::getInt32Ty(ctx);
  FunctionType *MemcpyWrapperFTy =
      FunctionType::get(retTy, {dstTy, srcTy, sizeTy, kindTy}, false);
  MemcpyWrapper = M.getOrInsertFunction("cudaMemcpyWrapper", MemcpyWrapperFTy);

  // declare cudaKernelLaunchPrepare()
  // resuse sizeTy and retTy for int64_t and int32_t
  FunctionType *KernelLaunchPrepareFTy =
      FunctionType::get(retTy, {sizeTy, retTy, sizeTy, retTy}, false);
  KernelLaunchPrepare =
      M.getOrInsertFunction("cudaKernelLaunchPrepare", KernelLaunchPrepareFTy);

  // declare cudaFreeWrapper()
  FunctionType *FreeFTy = FunctionType::get(retTy, {dstTy}, false);
  FreeWrapper = M.getOrInsertFunction("cudaFreeWrapper", FreeFTy);

  return true;
}

bool WrapperPass::runOnModule(Module &M) {
  Module::FunctionListType &Funcs = M.getFunctionList();
  for (auto ft = Funcs.begin(); ft != Funcs.end(); ft++) {
    Function &F = *ft;
    if (F.isIntrinsic() || F.isDeclaration()) continue;

    SmallVector<CallInst *, 4> ToBeRemoved;
    for (auto it = inst_begin(F); it != inst_end(F); it++) {
      Instruction *I = &*it;
      auto CI = dyn_cast<CallInst>(I);

      if (!CI) continue;

      auto Callee = CI->getCalledFunction();
      if (!Callee) continue;

      auto name = Callee->getName();

      if (name == "cudaMalloc") {
        replaceMalloc(CI);
        ToBeRemoved.push_back(CI);
      } else if (name == "cudaMemcpy") {
        replaceMemcpy(CI);
        ToBeRemoved.push_back(CI);
      } else if (name == "cudaFree") {
        replaceFree(CI);
        ToBeRemoved.push_back(CI);
      } else if (name == "__cudaPushCallConfiguration") {
        addKernelLaunchPrepare(CI);
      }
    }
    for (auto CI : ToBeRemoved) CI->eraseFromParent();
  }
  return true;
}

char WrapperPass::ID = 0;

#if 1
static RegisterPass<WrapperPass> X("WP", "WrapperPass", false, false);

#else

static void registerWP(const PassManagerBuilder &,
                       legacy::PassManagerBase &PM) {
  PM.add(new WrapperPass());
}

// Use EP_OptimizerLast to make sure the pass is run after all other
// optimization passes, such that the debug data is not removed by others
static RegisterStandardPasses RegisterMyPass(
    PassManagerBuilder::EP_OptimizerLast, registerWP);
#endif