# Sokoban solver

Push-optimal Sokoban solver in a single `main.ts` (Node 22, no dependencies).
Reads `/input/<id>/level.txt`, writes `/output/<id>/solution.txt` for all five
instances in one run.

## Algorithm

**Rooms first.** A box pushed out of a room through its door can never come back
(it plugs the only door and strands the player inside), so each room is an
independent problem. The board is split by cutting the dead floor squares of the
hallway row and flood-filling what is left; each part holding boxes is a room.
Rooms are solved in order, the player's position carrying over, and the other
rooms' boxes act as walls while a room is solved. A level's push count is the sum
of its rooms'.

**A\* over pushes.** Inside a room the search runs over pushes, not steps. A state
is the sorted box squares plus the player's reachable region (named by its lowest
square). The cost so far is pushes made; the estimate is the cheapest matching of
boxes to distinct goals, where one box-to-goal cost is the fewest pushes that box
alone would need, found by searching backwards from the goals. That estimate
never overestimates, so the first solved state popped is push-optimal.

**Pruning.** A push is dropped when it lands on a square from which no goal is
reachable, when it freezes a 2x2 block of walls and boxes holding a box off its
goal, or when it would carry a box out of its room. A room counts as solved only
when every box is on a goal and the player can still reach the door — a room whose
cheapest solution strands the player inside is no solution, since the player must
walk on to the next room (the last room needs no exit). The search is budgeted by
states counted, shared by a whole instance (300,000 states, calibrated so five
instances finish in ~30 s worst case), never by the clock, so two runs write
identical bytes; a plain breadth-first search over pushes is the fallback if A\*
exhausts its budget.

**Moves.** Each push in the found path is preceded by the player's shortest walk
to the push square (breadth-first search around walls and boxes); walk steps are
lower-case, pushes upper-case.

## Results (practice set)

| instance | pushes | P\* | points |
|----------|--------|-----|--------|
| i001 | 13 | 13 | 1000 |
| i002 | 16 | 16 | 1000 |
| i003 | 29 | 29 | 1000 |
| i004 | 47 | 47 | 1000 |
| i005 | 60 | 60 | 1000 |

Total score **1,000,000** (house baseline plain-astar: 200,000). Verified on the
practice set and on 10 freshly generated sets (all 50 instances push-optimal).
Runtime ~0.3 s for a set of five, far inside the 60 s budget; output is
byte-identical across runs.
