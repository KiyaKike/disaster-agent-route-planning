"""Black-and-white footprint-center map for the two-step diagnostic."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output
    data = json.loads((out/'geometry.json').read_text())
    xy = np.array(data['xy_m'])
    local = xy-xy.min(axis=0)
    positive = [2,6,8]
    fig, ax = plt.subplots(figsize=(6.4,5.8))
    for i, (x,y) in enumerate(local):
        ax.scatter(x,y,s=55,c='black' if i in positive else 'white',edgecolors='black',zorder=3)
        ax.annotate(f'B{i+1:02d}',(x,y),xytext=(5,6),textcoords='offset points',fontsize=10)
    for a,b,offset in [(2,6,(8,0)),(2,8,(-42,-2)),(6,8,(0,-14))]:
        line=local[[a,b]]
        ax.plot(line[:,0],line[:,1],color='black',ls='--',lw=.8,zorder=1)
        ax.annotate(f'{np.linalg.norm(xy[a]-xy[b]):.1f} m',line.mean(axis=0),xytext=offset,
                    textcoords='offset points',fontsize=9)
    ax.scatter([],[],c='black',edgecolors='black',label='UAS-view positive label')
    ax.scatter([],[],c='white',edgecolors='black',label='UAS-view negative label')
    ax.legend(loc='lower left',fontsize=8,frameon=False)
    ax.set(xlabel='Local easting (m)',ylabel='Local northing (m)',
           xlim=(-8,121),ylim=(-10,115),title='Boone: 12 building centers')
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(out/'building_map.png',dpi=180,facecolor='white')
    fig.savefig(out/'building_map.svg',facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
