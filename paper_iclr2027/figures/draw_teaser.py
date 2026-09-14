"""Scientific-figure-making house style: actual cache path and measured curves.

Run beside teaser_data.json. PDF/SVG retain vector text; PNG is 300 dpi.
Dense OOMs are status annotations outside the numerical axes, never fake points.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D
from matplotlib import font_manager
HERE=Path(__file__).resolve().parent
DATA=json.loads((HERE/'teaser_data.json').read_text())
BLUE,BLUE2,PALE='#0F4D92','#3775BA','#EAF1F8'
GREEN,GPALE='#356C49','#DDF3DE'
RED,INK,GRAY,RULE='#B64342','#272727','#767676','#CFCECE'
fonts={f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':[f for f in ['Arial','Helvetica','DejaVu Sans'] if f in fonts],
    'font.size':15,'axes.labelsize':15,'axes.titlesize':16,'axes.linewidth':1.05,
    'axes.spines.top':False,'axes.spines.right':False,'xtick.labelsize':14,'ytick.labelsize':14,
    'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,
    'savefig.facecolor':'white','mathtext.fontset':'dejavusans'})

def label(ax,x,y,s,size=15,color=INK,weight='normal',ha='center',**kw):
    return ax.text(x,y,s,fontsize=size,color=color,fontweight=weight,ha=ha,va='center',linespacing=1.15,zorder=7,**kw)
def rect(ax,x,y,w,h,fc='white',ec=RULE,lw=1.05,z=2):
    p=Rectangle((x,y),w,h,facecolor=fc,edgecolor=ec,linewidth=lw,zorder=z); ax.add_patch(p); return p
def arrow(ax,points,color=GRAY,dashed=False,lw=1.45):
    p=MplPath(points,[MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1))
    ax.add_patch(FancyArrowPatch(path=p,arrowstyle='-|>',mutation_scale=11,color=color,linewidth=lw,
        linestyle=(0,(3.5,2.5)) if dashed else 'solid',zorder=5))
def glyphs(ax,x,y,w,h,color,count=7):
    for i in range(count): rect(ax,x+i*w/count,y,w/count-.8,h,color,'none',0,4)
def layers(ax,x,y,w,h,edge,fill):
    for dx,dy in [(1.8,1.6),(.9,.8),(0,0)]: rect(ax,x+dx,y+dy,w,h,fill,edge,1.05)

def schematic(ax):
    ax.set(xlim=(0,121),ylim=(0,96)); ax.set_axis_off()
    label(ax,1,92,'(a) Encode once; reuse across queries',16,weight='bold',ha='left')
    # One shared LLM split around a persistent residual-state cache.
    layers(ax,47,17,46,15,GRAY,'#F6F6F6')
    label(ax,70,27.4,'Semantic encoder',16,weight='bold')
    label(ax,70,21.4,r'Lower layers  $F_{0:j}$',14)
    layers(ax,47,63,46,15,BLUE2,PALE)
    label(ax,70,73.5,'Query-aware reader',16,BLUE,weight='bold')
    label(ax,70,67.5,r'Upper layers  $F_{j:L}$',14,BLUE)
    rect(ax,44,41,54,12,PALE,BLUE2,1.3)
    label(ax,71,50.1,r'Hidden cache at $j=12$',15,BLUE,weight='bold')
    for k in range(5):
        rect(ax,48+k*9.5,43,7.5,3.8,'white',BLUE2,.6)
        for row in range(2): glyphs(ax,48.5+k*9.5,43.5+row*1.4,7,1,BLUE2,4)
    # Offline document stream passes the lower layers once, producing cache states.
    glyphs(ax,3,16,26,3.3,GRAY)
    label(ax,16,11.6,'Document stream',13,GRAY)
    arrow(ax,[(29,17.7),(38,17.7),(38,20),(46.2,20)],GRAY)
    arrow(ax,[(60,34),(60,40.3)],BLUE2)
    label(ax,66,37.1,'Write once',13,BLUE2,ha='left')
    # Query selects chunk IDs using BM25; the retrieved values are hidden states.
    glyphs(ax,3,44.2,26,3.3,GREEN)
    label(ax,16,39.6,'Query stream',13,GREEN)
    arrow(ax,[(29,46),(43.3,46)],GREEN,True)
    label(ax,22,55.8,'BM25 lookup',13,GREEN)
    label(ax,36,50.1,'IDs',12,GREEN)
    # The query still goes through lower layers; its states bypass persistent storage.
    arrow(ax,[(33,45.8),(33,28),(46.2,28)],GREEN)
    arrow(ax,[(95.8,26),(110,26),(110,58),(88,58),(88,62.3)],GREEN)
    label(ax,115.1,42,'Query states',13,GREEN,rotation=90)
    arrow(ax,[(72,53.6),(72,62.2)],BLUE,False,1.8)
    label(ax,67.5,58.1,'Fetch',13,BLUE,ha='right')
    arrow(ax,[(70,80),(70,83.2)],BLUE)
    glyphs(ax,57,83.7,27,2.6,BLUE2)
    label(ax,88,85,'Output',12,BLUE,ha='left')
    label(ax,61,3.5,'Only selected document states are fetched online',13,GRAY)

def arrays(method,phase,metric):
    xs=[]; ys=[]; lower=[]; upper=[]
    for n in (8192,16384,32768,131072):
        v=DATA['data'][str(n)][method][phase]
        xs.append(n/1024)
        if v['status']=='OOM': ys.append(np.nan); lower.append(np.nan); upper.append(np.nan); continue
        scale=1000 if metric=='latency_ms' else 1
        ys.append(v[metric]/scale)
        if metric=='latency_ms':
            medians=np.array(v['process_medians_ms'])/scale
            lower.append(ys[-1]-min(medians)); upper.append(max(medians)-ys[-1])
        else: lower.append(0); upper.append(0)
    return np.array(xs),np.array(ys),np.array([lower,upper])

HANDLES=[Line2D([],[],color=BLUE,marker='o',lw=2,label='MidCache'),
         Line2D([],[],color=RED,marker='s',ls='--',lw=2,label='Dense')]

def trend(ax,phase,metric,title,ylim,yticks,whiskers=False):
    ax.set_xscale('log',base=2); ax.set_xlim(6.7,151)
    ax.set_xticks([8,16,32,128],['8k','16k','32k','128k'])
    ax.set_ylim(*ylim); ax.set_yticks(yticks)
    ax.set_xlabel('Source length',labelpad=4)
    ax.set_title(title,loc='left',pad=38,fontweight='bold')
    ax.grid(axis='y',color='#E6E6E6',linewidth=.65,zorder=0)
    for name,col,mark,ls in [('dense',RED,'s','--'),('comem',BLUE,'o','-')]:
        x,y,err=arrays(name,phase,metric)
        ax.plot(x,y,color=col,marker=mark,markersize=5.7,lw=2.2,linestyle=ls,zorder=5)
        if whiskers and metric=='latency_ms':
            ax.errorbar(x,y,yerr=err,fmt='none',ecolor=col,elinewidth=1.1,capsize=3,zorder=4)
    # A categorical status row above the axes: these crosses have no numerical y-value.
    for x in (32,128):
        ax.plot(x,1.055,marker='x',ms=7,mew=1.5,color=RED,transform=ax.get_xaxis_transform(),clip_on=False)
    ax.text(.01,1.055,'OOM',transform=ax.transAxes,ha='left',va='center',fontsize=11.8,color=RED)
    if metric=='peak_GB':
        ax.axhline(28,color=GRAY,lw=1,ls=(0,(3,2)),zorder=1)
        ax.text(8,28.7,'28 GB cap',fontsize=11.8,color=GRAY,va='bottom')
    ax.tick_params(axis='both',length=3,pad=3)

def save(fig,stem):
    for ext in ('pdf','svg','png'): fig.savefig(HERE/f'{stem}.{ext}',dpi=300)
    plt.close(fig)

def main():
    # 11 inches exports at half scale in the 5.5-inch manuscript.
    fig=plt.figure(figsize=(11,3.9))
    schematic(fig.add_axes([.005,.035,.51,.94]))
    trend(fig.add_axes([.563,.235,.183,.55]),'ttft','latency_ms','(b) TTFT (s)',(0,3.15),[0,1,2,3])
    trend(fig.add_axes([.808,.235,.183,.55]),'ttft','peak_GB','(c) GPU peak (GB)',(16,30.5),[16,20,24,28])
    fig.legend(handles=HANDLES,loc='upper center',bbox_to_anchor=(.771,.905),ncol=2,fontsize=12,
               handlelength=1.7,columnspacing=1.4,borderaxespad=0)
    fig.text(.772,.028,'RTX 5090  |  shared adapter  |  28 GB cap',ha='center',fontsize=11.8,color=GRAY)
    save(fig,'teaser')
    # The complete scaling figure additionally charges document Write in E2E.
    fig=plt.figure(figsize=(11,3.55))
    for bounds,phase,metric,title,ylim,ticks in [
        ([.057,.255,.255,.5],'ttft','latency_ms','(a) TTFT (s)',(0,3.15),[0,1,2,3]),
        ([.394,.255,.255,.5],'e2e','latency_ms',r'(b) E2E$_{128}$ (s)',(0,14.4),[0,4,8,12]),
        ([.731,.255,.255,.5],'ttft','peak_GB','(c) GPU peak (GB)',(16,30.5),[16,20,24,28])]:
        trend(fig.add_axes(bounds),phase,metric,title,ylim,ticks,True)
    fig.legend(handles=HANDLES,loc='upper center',bbox_to_anchor=(.54,.888),ncol=2,fontsize=12,
               handlelength=1.9,columnspacing=1.7,borderaxespad=0)
    fig.text(.52,.025,'RTX 5090, shared adapter; E2E includes full document Write + 128 output tokens. Whiskers: process-median range.',
             ha='center',fontsize=11.8,color=GRAY)
    save(fig,'source_scaling')
    print('Exported teaser and source_scaling as PDF, SVG, and PNG.')

if __name__=='__main__': main()
