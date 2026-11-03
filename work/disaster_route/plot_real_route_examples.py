"""Post-execution scientific examples selected by the predeclared identity rank."""
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_real_xbd_routes import ROOT,OUT,SOURCE,read,inventory


def main():
    assert not (OUT/'completion_receipt.json').exists()
    selected=read(SOURCE/'selected.json');jobs,_,refs=inventory();routes=read(OUT/'routes.json')
    examples=sorted([s for s in selected if s['selection_rank']==1],key=lambda s:s['event'])
    fig,axes=plt.subplots(3,4,figsize=(13,10),constrained_layout=True)
    for row,s in enumerate(examples):
        sid=s['sample_id'];pair=[np.asarray(Image.open(next(j['path'] for j in jobs[sid]['images'] if j['mode']==m)).resize((512,512),Image.Resampling.BOX)) for m in ('pre','post')]
        truth=np.asarray(Image.open(refs[sid]['reference_file']).resize((512,512),Image.Resampling.NEAREST));target=np.isin(truth,[3,4]);overlay=np.zeros((512,512,4));overlay[target]=[.9,.15,.1,.65]
        axes[row,0].imshow(pair[0]);axes[row,0].set_title('Real pre-disaster image')
        axes[row,1].imshow(pair[1]);axes[row,1].imshow(overlay);axes[row,1].set_title('Real post + evaluator reference')
        for col,policy in [(2,'post_static'),(3,'online_neighbor')]:
            r=next(r for r in routes if r['sample_id']==sid and r['policy']==policy and r['budget_s_proxy']==240);state=r['state']
            ax=axes[row,col];ax.imshow(pair[1]);ax.imshow(overlay)
            nodes=[16,*state['visited'],16];xy=np.array([[i%4*128+64,i//4*128+64] if i!=16 else [256,511] for i in nodes])
            ax.plot(xy[:,0],xy[:,1],'-o',c='#00b6b0',ms=4,lw=1.7);ax.scatter(256,511,marker='*',s=85,c='black')
            ax.set_title(f"{policy}\n{len(state['visited'])} paid tiles, {state['elapsed_s_proxy']:.1f} proxy units")
        axes[row,0].set_ylabel(sid,fontsize=9)
        for ax in axes[row]:ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('REAL DATA | fixed first identity-ranked example per event | no favorable replacement\nRed = published severe reference for evaluation only;cyan = complete paid route,not a field flight path')
    fig.savefig(OUT/'figures/fixed_real_examples.png',dpi=170);plt.close(fig)
    print('FIXED_REAL_EXAMPLES_READY')


if __name__=='__main__':main()
