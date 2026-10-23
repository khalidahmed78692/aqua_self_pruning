import argparse, csv, os
from train.common import build, train, evaluate

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=80); ap.add_argument("--seed",type=int,default=7)
    ap.add_argument("--sparsity",type=float,default=0.9); ap.add_argument("--method",choices=["saliency","magnitude"],default="saliency")
    args=ap.parse_args(); os.makedirs("results",exist_ok=True)
    model,opt,Xtr,ytr,Xte,yte=build(args.seed)
    hist=train(model,opt,Xtr,ytr,epochs=args.epochs,seed=args.seed,pruning=(args.sparsity,args.method, int(args.epochs*0.2)))
    loss,acc=evaluate(model,Xte,yte)
    total=sum(l.mask.size for l in model.layers); active=sum(int(l.mask.sum()) for l in model.layers)
    path=f"results/pruned_{args.method}_{args.sparsity:.2f}_seed{args.seed}.csv"
    with open(path,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["epoch","train_loss","train_accuracy"]); w.writerows(hist)
    print(f"method={args.method} target_sparsity={args.sparsity:.3f} achieved_sparsity={1-active/total:.4f} active={active}/{total} test_loss={loss:.4f} test_accuracy={acc:.4f}")
if __name__=="__main__": main()
