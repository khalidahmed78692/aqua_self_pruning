import argparse, csv, os
import matplotlib.pyplot as plt
from train.common import build, train, evaluate

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-per-class", type=int, default=300)
    args=ap.parse_args()
    os.makedirs("results", exist_ok=True)
    model,opt,Xtr,ytr,Xte,yte=build(args.seed,args.n_per_class)
    hist=train(model,opt,Xtr,ytr,epochs=args.epochs,seed=args.seed)
    loss,acc=evaluate(model,Xte,yte)
    with open("results/baseline_history.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["epoch","train_loss","train_accuracy"]); w.writerows(hist)
    plt.figure(); plt.plot([h[0] for h in hist],[h[1] for h in hist]); plt.xlabel("Epoch"); plt.ylabel("Training loss"); plt.title("Dense baseline loss"); plt.tight_layout(); plt.savefig("results/baseline_loss.png"); plt.close()
    plt.figure(); plt.plot([h[0] for h in hist],[100*h[2] for h in hist]); plt.xlabel("Epoch"); plt.ylabel("Training accuracy (%)"); plt.title("Dense baseline accuracy"); plt.tight_layout(); plt.savefig("results/baseline_accuracy.png"); plt.close()
    print(f"seed={args.seed} test_loss={loss:.4f} test_accuracy={acc:.4f}")
if __name__=="__main__": main()
