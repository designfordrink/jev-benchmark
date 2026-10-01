from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np

BOARD_WIDTH=10
BOARD_HEIGHT=20

SHAPES={
    0: [[(0,0),(1,0),(0,1),(1,1)]],
    1: [[(0,0),(1,0),(2,0),(3,0)],[(0,0),(0,1),(0,2),(0,3)]],
    2: [[(1,0),(0,1),(1,1),(2,1),(1,2)]],
    3: [[(1,0),(2,0),(0,1),(1,1)] ,[(0,0),(0,1),(1,1),(1,2)]],
    4: [[(0,0),(1,0),(1,1),(2,1)] ,[(1,0),(0,1),(1,1),(0,2)]],
    5: [[(0,0),(0,1),(1,1),(2,1)] ,[(0,0),(1,0),(0,1),(0,2)],[(0,0),(1,0),(2,0),(2,1) ],[(1,0),(1,1),(0,2),(1,2)]],
    6: [[(2,0),(0,1),(1,1),(2,1)] ,[(0,0),(0,1),(0,2),(1,2)],[(0,0),(1,0),(2,0),(0,1)],[(0,0),(1,0),(1,1),(1,2)]],
}

@dataclass(frozen=True)
class TetrisPlacement:
    candidate_id:int
    rotation:int
    x:int
    y:int
    lines_cleared:int
    holes:int
    max_height:int
    aggregate_height:int
    bumpiness:int
    features:np.ndarray

class TetrisEnv:
    """Small deterministic Tetris environment used for candidate-selection research."""
    name="j04_tetris"
    def __init__(self,seed:int=0,*,width:int=BOARD_WIDTH,height:int=BOARD_HEIGHT):
        self.width=width; self.height=height; self.rng=np.random.default_rng(seed); self.seed=seed
        self.board=np.zeros((height,width),dtype=np.uint8); self.game_over=False; self.score=0; self.pieces=0; self.last_lines=0
        self.current_piece=0
    def reset(self,seed:int|None=None)->np.ndarray:
        if seed is not None: self.rng=np.random.default_rng(seed); self.seed=seed
        self.board.fill(0); self.game_over=False; self.score=0; self.pieces=0; self.last_lines=0; self.current_piece=int(self.rng.integers(0,7))
        return self.observation()
    def observation(self)->np.ndarray:
        heights=self.column_heights(self.board)
        holes=self.count_holes(self.board)
        return np.asarray([*heights,self.aggregate_height(heights),holes],dtype=np.float32)
    @staticmethod
    def column_heights(board:np.ndarray)->np.ndarray:
        heights=np.zeros(board.shape[1],dtype=np.int32)
        for x in range(board.shape[1]):
            filled=np.flatnonzero(board[:,x])
            heights[x]=0 if len(filled)==0 else board.shape[0]-int(filled[0])
        return heights
    @staticmethod
    def count_holes(board:np.ndarray)->int:
        holes=0
        for x in range(board.shape[1]):
            filled=np.flatnonzero(board[:,x])
            if len(filled):
                top=int(filled[0]); holes+=int(np.count_nonzero(board[top:,x]==0))
        return holes
    @staticmethod
    def aggregate_height(heights:np.ndarray)->int:return int(np.sum(heights))
    @staticmethod
    def bumpiness(heights:np.ndarray)->int:return int(np.sum(np.abs(np.diff(heights)))) if len(heights)>1 else 0
    def _cells(self,piece:int,rotation:int,x:int,y:int)->list[tuple[int,int]]:
        return [(x+dx,y+dy) for dx,dy in SHAPES[piece][rotation]]
    def _valid(self,cells:list[tuple[int,int]],board:np.ndarray|None=None)->bool:
        b=self.board if board is None else board
        return all(0<=x<self.width and 0<=y<self.height and b[y,x]==0 for x,y in cells)
    def legal_placements(self)->list[TetrisPlacement]:
        placements=[]; cid=0; seen=set()
        for rotation,shape in enumerate(SHAPES[self.current_piece]):
            max_dx=max(x for x,_ in shape); min_dx=min(x for x,_ in shape)
            for x in range(-min_dx,self.width-max_dx):
                y=0
                if not self._valid(self._cells(self.current_piece,rotation,x,y)): continue
                while self._valid(self._cells(self.current_piece,rotation,x,y+1)): y+=1
                cells=self._cells(self.current_piece,rotation,x,y)
                after=self.board.copy()
                for px,py in cells: after[py,px]=1
                full=np.all(after==1,axis=1); lines=int(full.sum()); after=after[~full]
                if lines:
                    after=np.vstack([np.zeros((lines,self.width),dtype=np.uint8),after])
                heights=self.column_heights(after); holes=self.count_holes(after); agg=self.aggregate_height(heights); bump=self.bumpiness(heights); maxh=int(heights.max())
                key=(rotation,x,y)
                if key in seen: continue
                seen.add(key)
                feat=np.asarray([x/self.width,y/self.height,lines/max(1,self.height),holes/self.width,maxh/self.height,agg/(self.width*self.height),bump/(self.width*self.height),*heights[:self.width]/self.height],dtype=np.float32)
                placements.append(TetrisPlacement(cid,rotation,x,y,lines,holes,maxh,agg,bump,feat)); cid+=1
        return placements
    def step(self,placement:TetrisPlacement)->tuple[np.ndarray,float,bool,dict]:
        if self.game_over: raise RuntimeError("game is over")
        legal=self.legal_placements()
        if not any((p.rotation,p.x,p.y)==(placement.rotation,placement.x,placement.y) for p in legal): raise ValueError("illegal Tetris placement")
        cells=self._cells(self.current_piece,placement.rotation,placement.x,placement.y)
        for px,py in cells: self.board[py,px]=1
        full=np.all(self.board==1,axis=1); lines=int(full.sum());
        if lines: 
            remaining=self.board[~full]; self.board=np.vstack([np.zeros((lines,self.width),dtype=np.uint8),remaining])
        self.last_lines=lines; self.score+=lines*lines*100+1; self.pieces+=1
        self.current_piece=int(self.rng.integers(0,7)); self.game_over=not bool(self.legal_placements())
        reward=float(lines*lines*100 + (0 if self.game_over else 1))
        return self.observation(),reward,self.game_over,{"lines_cleared":lines,"pieces":self.pieces}

    def run_episode(self,selector,*,seed:int|None=None,max_pieces:int=500)->dict:
        self.reset(seed); total_reward=0.; lines=0; legal_counts=[]; decisions=0; lat=[]
        while not self.game_over and self.pieces<max_pieces:
            legal=self.legal_placements(); legal_counts.append(len(legal))
            if not legal: self.game_over=True; break
            context=self.observation()
            import time
            t=time.perf_counter_ns(); idx,conf,_=selector.rank(context,legal); lat.append((time.perf_counter_ns()-t)/1000.0)
            chosen=legal[idx]; _,reward,done,info=self.step(chosen); total_reward+=reward; lines+=int(info["lines_cleared"]); decisions+=1
        return {"return":float(total_reward),"lines":lines,"pieces":self.pieces,"mean_legal_candidates":float(np.mean(legal_counts) if legal_counts else 0),"mean_latency_us":float(np.mean(lat) if lat else 0),"p95_latency_us":float(np.percentile(lat,95) if lat else 0),"decisions":decisions}
