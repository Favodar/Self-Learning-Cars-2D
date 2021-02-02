# Self-Learning-Cars-2D

#### Action space
```[a1, a2]```  
  
a1 = [0/1/2] = throttle [no/half/full]  
a2 = [0/1/2] = steer [left/straight/right]  
all action values are integers.

#### Observation space
```[[x1, y1], [x2, y2], [s2, r2]]```  
  
[x1, y1] = coordinates enemy in 2D euclidian space  
[x2, y2] = coordinates self in 2D euclidian space  
[s2, r2] = self speed, self rotation  
All observed values are floating point numbers.
