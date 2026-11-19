"""Native edit operations with exact coverage and RAM/SSD legality."""
import numpy as np

class NativeLegalEdits:
 def __init__(self,e,X):
  self.spec=np.asarray(e.spec,dtype=float)
  self.capacity=np.asarray(e.capacity,dtype=float)
  self.X=np.asarray(X,dtype=bool).copy()
  self._assert_legal()
 def _assert_legal(self):
  assert self.X.any(1).all()
  assert np.all(self.X.T@self.spec<=self.capacity+1e-7)
 def legal_add(self,k,n):
  if self.X[k,n]:return False
  free=self.capacity[n]-self.X[:,n]@self.spec
  return bool(np.all(free+1e-7>=self.spec[k]))
 def legal_remove(self,k,n):
  return bool(self.X[k,n] and self.X[k].sum()>1)
 def legal_relocate(self,k,from_node,to_node):
  if from_node==to_node or not self.X[k,from_node] or self.X[k,to_node]:return False
  # The move is atomic: the stage is never uncovered and freed capacity is
  # accounted for exactly. Only the destination can create a capacity issue.
  free=self.capacity[to_node]-self.X[:,to_node]@self.spec
  return bool(np.all(free+1e-7>=self.spec[k]))
 def apply(self,action):
  kind,k,n,other=action
  if kind=='ADD':
   if not self.legal_add(k,n):raise ValueError('Illegal ADD')
   self.X[k,n]=True
  elif kind=='REMOVE':
   if not self.legal_remove(k,n):raise ValueError('Illegal REMOVE')
   self.X[k,n]=False
  elif kind=='RELOCATE':
   if not self.legal_relocate(k,n,other):raise ValueError('Illegal RELOCATE')
   self.X[k,n]=False;self.X[k,other]=True
  else:raise ValueError('Unknown edit action')
  self._assert_legal()
  return self.X.copy()
