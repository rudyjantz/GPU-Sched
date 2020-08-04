#ifndef _GPU_BEACON_H_
#define _GPU_BEACON_H_

#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/NoFolder.h>
#include <llvm/Pass.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils.h>

using namespace llvm;

class WrapperPass : public ModulePass {
 private:
  FunctionCallee MallocWrapper;
  FunctionCallee MemcpyWrapper;
  FunctionCallee MemcpyToSymbolWrapper;
  FunctionCallee KernelLaunchPrepare;
  FunctionCallee FreeWrapper;

  void replaceMalloc(CallInst *CI);
  void replaceMemcpy(CallInst *CI);
  void replaceMemcpyToSymbol(CallInst *CI);
  void replaceFree(CallInst *CI);
  void addKernelLaunchPrepare(CallInst *CI);

 public:
  static char ID;

  WrapperPass() : ModulePass(ID) {}
  virtual bool doInitialization (Module &) override;
  virtual bool runOnModule(Module &M) override;
};

#endif