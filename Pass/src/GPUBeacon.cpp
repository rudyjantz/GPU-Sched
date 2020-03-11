#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
using namespace llvm;

namespace {
struct GPUBeaconPass : public ModulePass {
  static char ID;
  GPUBeaconPass() : ModulePass(ID) {}

  virtual bool runOnModule(Module &M) {
    errs() << "I saw a module called " << M.getName() << "!\n";
    return false;
  }
};
}  // namespace

char GPUBeaconPass::ID = 0;

// Automatically enable the pass.
// http://adriansampson.net/blog/clangpass.html
static void registerSkeletonPass(const PassManagerBuilder &,
                                 legacy::PassManagerBase &PM) {
  PM.add(new GPUBeaconPass());
}
static RegisterStandardPasses RegisterMyPass(
    PassManagerBuilder::EP_EarlyAsPossible, registerSkeletonPass);
