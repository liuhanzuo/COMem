"""Editable publication schematic of the CoMem residual interface.

scientific-figure-making: thin rules, semantic blue/green, vector text.
Labels become 7.5-8.5 pt at the manuscript's 5.5-inch width. Token glyphs
are illustrative and encode no experimental quantities.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.path import Path as MplPath
OUT=Path(__file__).resolve().parent
INK,GRAY,RULE='#272727','#767676','#CFCECE'
BLUE,MID,PALE='#0F4D92','#3775BA','#EAF1F8'
GREEN,GD,GM='#DDF3DE','#356C49','#86B797'
available={f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':[f for f in ['Arial','Helvetica','DejaVu Sans'] if f in available],'font.size':16,'mathtext.fontset':'dejavusans','svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
fig=plt.figure(figsize=(11,5.35))
ax=fig.add_axes([.008,.009,.984,.982])
ax.set(xlim=(0,100),ylim=(0,48.5)); ax.set_axis_off()
def label(x,y,s,size=16,color=INK,weight='normal',ha='center',**kw):
    return ax.text(x,y,s,fontsize=size,color=color,fontweight=weight,ha=ha,va='center',linespacing=1.15,zorder=7,**kw)
def rect(x,y,w,h,fc='white',ec=RULE,lw=1.05,z=2):
    p=Rectangle((x,y),w,h,facecolor=fc,edgecolor=ec,linewidth=lw,zorder=z); ax.add_patch(p); return p
def arrow(points,color=GRAY,lw=1.25,dashed=False):
    path=MplPath(points,[MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1))
    ax.add_patch(FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=12,color=color,linewidth=lw,zorder=5,linestyle=(0,(3.5,2.8)) if dashed else 'solid'))
def stack(x,y,w,h,fc=PALE,ec=MID):
    for dx,dy in [(.8,.95),(.4,.475),(0,0)]: rect(x+dx,y+dy,w,h,fc,ec)
def matrix(x,y,w,h,rows=4,cols=6,color=MID):
    for r in range(rows):
        for c in range(cols): rect(x+c*w/cols,y+r*h/rows,w/cols-.13,h/rows-.13,color,'none',0,4)
def tokens(x,y,w,h,count=6,color='#B8BDC2'):
    for c in range(count): rect(x+c*w/count,y,w/count-.18,h,color,'none',0,4)
label(1,47.1,'(a) Encode and cache',17,weight='bold',ha='left')
label(42,47.1,'(b) Read with the query',17,weight='bold',ha='left')
ax.plot([39.7,39.7],[3.2,44],color=RULE,lw=.9,zorder=0)
# Write chunks independently with shared lower layers.
label(19.3,43.1,'Persistent residual memory',16,BLUE,'bold')
rect(1.5,31,35.5,10.15,'#F5F8FC',RULE,1)
for x,idx in [(4.,'1'),(16.,'2'),(28.,'n')]:
    label(x+3.25,39.55,rf'$H_{idx}$',17,BLUE); matrix(x,34.1,6.5,4.2)
label(19.3,32.35,r'$H_i\in\mathbb{R}^{|x_i|\times d}$: one vector per token',15,BLUE)
for x,idx in [(2.5,'1'),(15.1,'2'),(27.7,'n')]:
    cx=x+4.25
    tokens(x,8.5,8.5,2.1); label(cx,6.65,rf'$x_{idx}$',17)
    arrow([(cx,11),(cx,18.4)])
    stack(x,18.8,8.5,5.6); label(cx,21.6,r'$F_{0:j}$',17,BLUE)
    arrow([(cx,25.8),(cx,30.55)],MID)
label(19.3,15.4,'Lower layers: semantic encoder',15,GRAY,bbox=dict(facecolor='white',edgecolor='none',pad=.8))
label(19.3,3.65,'Document chunks encoded independently',15,GRAY)
# Query selects IDs and separately produces its lower-layer states.
rect(43.2,11.4,18.5,6.1,'white',GRAY)
label(52.45,15.55,'Select',16,weight='bold'); label(52.45,13.25,r'BM25 top-$k$',15,GRAY)
arrow([(43.,14.45),(38.35,14.45),(38.35,28),(35.5,28),(35.5,30.65)],GRAY,dashed=True)
label(41.6,20.15,'Chunk IDs',15,GRAY,ha='left')
tokens(83,6,11.5,2,color=GM); label(88.75,3.65,r'Query $q$',16)
arrow([(82.5,7),(52.45,7),(52.45,10.95)],GRAY)
arrow([(88.75,8.4),(88.75,10.85)],GD)
stack(81,11.3,15.5,6,GREEN,GD)
label(88.75,15.35,r'Write $F_{0:j}$',16,GD); label(88.75,13.05,'Query + sink',15,GD)
arrow([(92,18.6),(92,24.1)],GD)
# Horizontal pack preserves sink, selected residuals, then query order.
rect(53,24.5,43.5,5,'white',RULE,1)
for x,w,s,fill,ink,n in [(53.5,5,'Sink',GREEN,GD,1),(59.1,9.1,r'$H_{s_1}$',PALE,BLUE,5),(74,9.1,r'$H_{s_m}$',PALE,BLUE,5),(84,12,r'$h_j(q)$',GREEN,GD,6)]:
    rect(x,24.95,w,4.1,fill,'none',0,3)
    label(x+w/2,27.7,s,15,ink)
    tokens(x+.45,25.45,w-.75,.65,n,GM if ink==GD else MID)
label(71.15,27,r'$\cdots$',20,BLUE)
label(69.4,22.55,r'Read pack $P_j$; document order',15,GRAY)
arrow([(37.4,37.4),(44,37.4),(44,32),(70.7,32),(70.7,29.85)],MID,1.5)
label(53.2,33.45,'Fetch residuals',15,BLUE,bbox=dict(facecolor='white',edgecolor='none',pad=.35))
# Continue at j; LoRA marks the principal configuration (optional in caption).
arrow([(82.1,29.85),(82.1,34.65)],BLUE,1.5)
stack(58,35.05,37,6.35,PALE,MID)
label(76.5,39.4,r'Upper-layer reader $[j{:}L)$',16,BLUE)
label(70.7,36.95,'Causal attention',15,BLUE)
rect(85,35.9,9,2.35,'#F6CFCB','none',0,4); label(89.5,37.1,'LoRA',15)
arrow([(78,42.55),(78,43.45)],BLUE,1.35); label(78,44.7,'Next-token logits',15)
for suffix in ('pdf','svg','png'): fig.savefig(OUT/f'architecture.{suffix}',dpi=300)
plt.close(fig)
print('Wrote PDF, SVG, PNG')
