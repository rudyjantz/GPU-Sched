#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/Pass.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>

using namespace llvm;

namespace {

const std::string getDemangledFunctionName(const Function &F) {
  std::string name = F.getName().str();
  return demangle(name);
}

const std::string getDemangledFunctionName(const Function *F) {
  return getDemangledFunctionName(*F);
}

struct GPUBeaconPass : public ModulePass {
  static char ID;
  GPUBeaconPass() : ModulePass(ID) {}

  void runOnFunction(Function &F) {
    // looking for __cudaPopCallConfiguration CallInst
    for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
      CallInst *CI = dyn_cast<CallInst>(&*I);
      if (!CI) continue;
      Function *Callee = CI->getCalledFunction();
      auto name = getDemangledFunctionName(Callee);
      if (name != "__cudaPopCallConfiguration") continue;
      dbgs() << "find a PopCallConfiguration: " << *I << "\n";

      auto gridDim = CI->getArgOperand(0);
      auto blockDim = CI->getArgOperand(1);
      dbgs() << "gridDim: " << *gridDim << ", blockDim: " << *blockDim << "\n";
    }
  }

  virtual bool runOnModule(Module &M) {
    Module::FunctionListType &funcs = M.getFunctionList();
    for (auto it = funcs.begin(); it != funcs.end(); it++) {
      Function &F = *it;
      if (F.isIntrinsic() || F.isDeclaration()) continue;

      auto name = getDemangledFunctionName(F);

      if (name.find("__device_stub_") == std::string::npos) continue;

      dbgs() << "Find a device stub code: " << name << "\n";
      runOnFunction(F);

    }  // end of module loop
    return false;
  }
};
}  // namespace

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
