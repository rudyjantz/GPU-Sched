

#include "GPUBeacon.h"

#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>

#include "CUDAInfo.h"

using namespace llvm;

bool GPUBeaconPass::runOnModule(Module &M) {
  CUDATaskBuilder CTB(M);
  CTB.build(M);

  return false;
}

char GPUBeaconPass::ID = 0;

static RegisterPass<GPUBeaconPass> X("GB", "GPUBeacon", false, false);

// Automatically enable the pass.
// http://adriansampson.net/blog/clangpass.html
// static void registerGPUBeaconPass(const PassManagerBuilder &,
//                                   legacy::PassManagerBase &PM) {
//   PM.add(new GPUBeaconPass());
// }
// static RegisterStandardPasses RegisterMyPass(
//     PassManagerBuilder::EP_EarlyAsPossible, registerGPUBeaconPass);
