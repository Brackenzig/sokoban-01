// Sokoban solver: push-optimal, room by room.
//
// Idea. A level is a hallway with one to four small rooms off it, and a box that
// leaves its room through the door can never come back (it plugs the only door and
// strands the player inside), so each room is solved on its own and the answers are
// stitched together. Inside a room the search runs over PUSHES, not steps: a state
// is the boxes' squares plus the player's reachable region, and A* orders states by
// pushes made plus a lower bound on the pushes still needed. The bound is the
// cheapest way to match every box to a different goal, where the cost of sending one
// box to one goal is the fewest pushes that box alone would need (found by searching
// backwards from the goals). That bound never overestimates, so the first solved
// state A* pops is push-optimal. Two deadlocks are pruned: a box pushed onto a
// square from which no goal is reachable, and a box frozen inside a 2x2 block of
// walls and boxes with a box off its goal. A room counts as solved only when every
// box is on a goal and the player can still reach the door: a room whose cheapest
// solution strands the player inside is no solution, since the player must walk on
// to the next room. The search is budgeted by states counted, shared by a whole
// instance, never by the clock, so two runs always write the same bytes.
//
// Local run: MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const IN_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/**
 * Work budget, counted in states generated (never the clock: runs must agree).
 * One instance may spend this many states, shared by its rooms and by the A* and
 * BFS fallback, so a hard instance can never starve the others or blow the 60 s wall:
 * a plain push search runs ~50,000 states/s, so five instances at this budget
 * finish in ~30 s worst case.
 */
const PER_INSTANCE_BUDGET = 300_000;

/** The states an instance has left to spend. Shared, never refreshed. */
interface Budget {
  remaining: number;
}

const DIRS = [
  { dx: 0, dy: -1, walk: 'u', push: 'U' },
  { dx: 0, dy: 1, walk: 'd', push: 'D' },
  { dx: -1, dy: 0, walk: 'l', push: 'L' },
  { dx: 1, dy: 0, walk: 'r', push: 'R' },
] as const;

interface Board {
  w: number;
  h: number;
  /** 1 = wall. The outer frame is forced to wall, and short rows pad to wall. */
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

function parseBoard(text: string): Board {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x]!;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

/**
 * For each goal, the fewest pushes that would carry one box alone from any square
 * to that goal. Searched backwards from the goal: a box at x may have come from
 * x - d when the square it was pushed from (x - d) and the square the player stood
 * on (x - 2d) are both open floor. Other boxes are ignored, so every entry is a
 * lower bound on the true cost. -1 means unreachable.
 */
function pushDistances(w: number, wall: Uint8Array, goals: number[]): Int32Array[] {
  const size = wall.length;
  const tables: Int32Array[] = [];
  for (const g of goals) {
    const dist = new Int32Array(size).fill(-1);
    dist[g] = 0;
    const queue: number[] = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const x = queue[qi]!;
      for (let k = 0; k < 4; k++) {
        const d = DIRS[k]!.dy * w + DIRS[k]!.dx;
        const y = x - d;
        const behind = y - d;
        if (y < 0 || y >= size || behind < 0 || behind >= size) continue;
        if (wall[y] || wall[behind]) continue;
        if (dist[y] >= 0) continue;
        dist[y] = dist[x]! + 1;
        queue.push(y);
      }
    }
    tables.push(dist);
  }
  return tables;
}

/** Every permutation of 0..n-1 (Heap's method). n is at most 4 here. */
function allPermutations(n: number): number[][] {
  const out: number[][] = [];
  const a = [...Array(n).keys()];
  const gen = (k: number): void => {
    if (k === 1) {
      out.push(a.slice());
      return;
    }
    for (let i = 0; i < k; i++) {
      gen(k - 1);
      const j = k % 2 === 1 ? 0 : i;
      const tmp = a[j]!;
      a[j] = a[k - 1]!;
      a[k - 1] = tmp;
    }
  };
  gen(n);
  return out;
}

/**
 * The squares the player reaches from `from` without stepping on a wall or a box.
 * `seen`/`stamp` mark the region, `stack` is scratch. Returns the region's lowest
 * square, which names the region: two regions never share a lowest square.
 */
function floodRegion(
  w: number,
  wall: Uint8Array,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    const col = p % w;
    let q = p - w;
    if (q >= 0 && wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
      seen[q] = stamp;
      stack[n++] = q;
    }
    q = p + w;
    if (q < wall.length && wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
      seen[q] = stamp;
      stack[n++] = q;
    }
    if (col > 0) {
      q = p - 1;
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
    if (col < w - 1) {
      q = p + 1;
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/**
 * True when the box just pushed to `t` is frozen inside a 2x2 block of walls and
 * boxes that holds a box off its goal. `occupied` already reflects the new boxes.
 */
function isFrozen(
  w: number,
  wall: Uint8Array,
  occupied: Uint8Array,
  goalSet: Uint8Array,
  t: number,
): boolean {
  const size = wall.length;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (a < 0 || d >= size) continue;
    if (!wall[a] && !occupied[a]) continue;
    if (!wall[b] && !occupied[b]) continue;
    if (!wall[c] && !occupied[c]) continue;
    if (!wall[d] && !occupied[d]) continue;
    if (
      (occupied[a] && !goalSet[a]) ||
      (occupied[b] && !goalSet[b]) ||
      (occupied[c] && !goalSet[c]) ||
      (occupied[d] && !goalSet[d])
    ) {
      return true;
    }
  }
  return false;
}

/** The shortest walk from `from` to `to` around walls and boxes, lower-case letters. */
function walkMoves(
  w: number,
  wall: Uint8Array,
  occupied: Uint8Array,
  from: number,
  to: number,
): string {
  if (from === to) return '';
  const size = wall.length;
  const prev = new Int32Array(size).fill(-1);
  const how = new Int32Array(size).fill(-1);
  const queue: number[] = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    const col = p % w;
    for (let k = 0; k < 4; k++) {
      const dir = DIRS[k]!;
      if (dir.dx === -1 && col === 0) continue;
      if (dir.dx === 1 && col === w - 1) continue;
      const q = p + dir.dy * w + dir.dx;
      if (q < 0 || q >= size) continue;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = k;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walk: target unreachable');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = DIRS[how[p]!]!.walk + s;
  return s;
}

/** The player's square after playing `moves` (case aside) from `from`. */
function endPosition(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves) {
    const c = ch.toLowerCase();
    if (c === 'u') p -= w;
    else if (c === 'd') p += w;
    else if (c === 'l') p -= 1;
    else p += 1;
  }
  return p;
}

interface SubLevel {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
  /** Squares of this room; a push must land inside it. Null for a whole-level search. */
  roomMask: Uint8Array | null;
  /**
   * The door squares the player must still reach when the boxes are solved, so the
   * player is not stranded inside the room. Null means no exit is needed (the last
   * room, or a whole-level search).
   */
  doors: number[] | null;
}

type SearchOutcome =
  | { status: 'solved'; pushes: number; moves: string }
  | { status: 'capped' }
  | { status: 'dead' };

/**
 * Push-optimal solve of one sub-level by A* over pushes (useHeuristic false runs
 * plain breadth-first search). A state is the sorted boxes plus the player's
 * reachable region, named by its lowest square. The estimate is the cheapest
 * matching of boxes to distinct goals by the backward push distances, or the sum of
 * each box's nearest-goal distance when there are too many boxes to match by
 * permutation. Both are lower bounds, so the first solved state popped is optimal.
 * Pushes are pruned when they land on a square no goal is reachable from, freeze a
 * 2x2 block, or (in a room) leave the room. `budget` bounds the work, shared with
 * every other search this instance runs. A state counts as solved only when every
 * box is on a goal and, when doors are given, the player can still reach a door:
 * a room whose cheapest solution strands the player inside is no solution at all.
 */
function searchPushes(sub: SubLevel, budget: Budget, useHeuristic: boolean): SearchOutcome {
  const { w, h, wall, goals, boxes: startBoxes, player: startPlayer, roomMask, doors } = sub;
  const size = w * h;
  const n = goals.length;
  const goalSet = new Uint8Array(size);
  for (const g of goals) goalSet[g] = 1;
  const distTables = pushDistances(w, wall, goals);
  const live = new Uint8Array(size);
  for (let s = 0; s < size; s++) {
    for (let gi = 0; gi < n; gi++) {
      if (distTables[gi]![s]! >= 0) {
        live[s] = 1;
        break;
      }
    }
  }
  if (startBoxes.some((b) => !live[b])) return { status: 'dead' };

  const perms = useHeuristic && n <= 5 ? allPermutations(n) : null;
  const estimate = (bs: number[]): number => {
    if (!useHeuristic) return 0;
    if (perms) {
      let best = Infinity;
      for (const p of perms) {
        let sum = 0;
        let ok = true;
        for (let i = 0; i < n; i++) {
          const d = distTables[p[i]!]![bs[i]!]!;
          if (d < 0) {
            ok = false;
            break;
          }
          sum += d;
        }
        if (ok && sum < best) best = sum;
      }
      return best;
    }
    let sum = 0;
    for (const b of bs) {
      let m = Infinity;
      for (let gi = 0; gi < n; gi++) {
        const d = distTables[gi]![b]!;
        if (d >= 0 && d < m) m = d;
      }
      if (m === Infinity) return Infinity;
      sum += m;
    }
    return sum;
  };

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const regionMark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;

  const solved = (bs: number[]): boolean => bs.every((b) => goalSet[b] === 1);
  const keyOf = (bs: number[], low: number): string => low + '|' + bs.join(',');

  // The states, in the order found.
  const stateBoxes: number[][] = [startBoxes.slice()];
  const statePlayer: number[] = [startPlayer];
  const stateParent: number[] = [-1];
  const statePushFrom: number[] = [-1];
  const statePushTo: number[] = [-1];
  const statePushDir: number[] = [-1];
  const stateG: number[] = [0];
  const stateLow: number[] = [0];
  const stateKey: string[] = [''];
  const bestG = new Map<string, number>();

  for (const b of startBoxes) occupied[b] = 1;
  stateLow[0] = floodRegion(w, wall, occupied, startPlayer, seen, ++stamp, stack);
  for (const b of startBoxes) occupied[b] = 0;
  stateKey[0] = keyOf(startBoxes, stateLow[0]!);
  const h0 = estimate(startBoxes);
  if (h0 === Infinity) return { status: 'dead' };
  if (solved(startBoxes)) return { status: 'solved', pushes: 0, moves: '' };
  bestG.set(stateKey[0]!, 0);

  // Open list: a binary heap on [f, g, seq, state], smallest f first, then deepest,
  // then earliest found, so ties break the same on every run.
  const heap: [number, number, number, number][] = [];
  const heapLess = (a: [number, number, number, number], b: [number, number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!heapLess(heap[i]!, heap[p]!)) break;
      const tmp = heap[i]!;
      heap[i] = heap[p]!;
      heap[p] = tmp;
      i = p;
    }
  };
  const heapPop = (): [number, number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && heapLess(heap[c + 1]!, heap[c]!)) c++;
        if (!heapLess(heap[c]!, heap[i]!)) break;
        const tmp = heap[i]!;
        heap[i] = heap[c]!;
        heap[c] = tmp;
        i = c;
      }
    }
    return top;
  };

  let seq = 0;
  heapPush([h0, 0, seq++, 0]);

  while (heap.length > 0) {
    const entry = heapPop();
    const g = entry[1];
    const s = entry[3];
    if (bestG.get(stateKey[s]!) !== g) continue; // a better route to this state exists
    const boxes = stateBoxes[s]!;
    if (solved(boxes)) {
      let isGoal = true;
      if (doors !== null && doors.length > 0) {
        // The player must still reach a door, or it is stranded in the room.
        for (const b of boxes) occupied[b] = 1;
        floodRegion(w, wall, occupied, statePlayer[s]!, seen, ++stamp, stack);
        const exitStamp = stamp;
        for (const b of boxes) occupied[b] = 0;
        isGoal = doors.some((d) => seen[d] === exitStamp);
      }
      if (isGoal) {
      // Rebuild the moves: walk to each push, then push.
      const pushes: { from: number; to: number; dir: number }[] = [];
      for (let t = s; t > 0; t = stateParent[t]!) {
        pushes.push({ from: statePushFrom[t]!, to: statePushTo[t]!, dir: statePushDir[t]! });
      }
      pushes.reverse();
      let moves = '';
      let player = startPlayer;
      const cur = startBoxes.slice();
      for (const b of cur) occupied[b] = 1;
      for (const push of pushes) {
        const dir = DIRS[push.dir]!;
        const d = dir.dy * w + dir.dx;
        moves += walkMoves(w, wall, occupied, player, push.from - d);
        moves += dir.push;
        occupied[push.from] = 0;
        occupied[push.to] = 1;
        const bi = cur.indexOf(push.from);
        cur[bi] = push.to;
        cur.sort((a, b2) => a - b2);
        player = push.from;
      }
      for (const b of cur) occupied[b] = 0;
      return { status: 'solved', pushes: pushes.length, moves };
      }
    }
    for (const b of boxes) occupied[b] = 1;
    floodRegion(w, wall, occupied, statePlayer[s]!, seen, ++stamp, stack);
    const regionStamp = stamp;
    for (let bi = 0; bi < n; bi++) {
      const b = boxes[bi]!;
      for (let k = 0; k < 4; k++) {
        const dir = DIRS[k]!;
        const d = dir.dy * w + dir.dx;
        const dest = b + d;
        const behind = b - d;
        if (behind < 0 || behind >= size) continue;
        if (seen[behind] !== regionStamp) continue; // player cannot reach the push square
        if (wall[dest] || occupied[dest]) continue; // destination blocked
        if (!live[dest]) continue; // a box there never reaches a goal
        if (roomMask && !roomMask[dest]) continue; // a box leaving its room never returns
        occupied[b] = 0;
        occupied[dest] = 1;
        const frozen = isFrozen(w, wall, occupied, goalSet, dest);
        let next: number[] | null = null;
        let newLow = -1;
        let newKey = '';
        if (!frozen) {
          next = boxes.slice();
          next[bi] = dest;
          next.sort((a, b2) => a - b2);
          newLow = floodRegion(w, wall, occupied, b, regionMark, ++stamp, stack);
          newKey = keyOf(next, newLow);
        }
        occupied[dest] = 0;
        occupied[b] = 1;
        if (frozen || next === null) continue;
        const prevBest = bestG.get(newKey);
        if (prevBest !== undefined && prevBest <= g + 1) continue;
        const hh = estimate(next);
        if (hh === Infinity) continue; // no way to match the boxes to distinct goals
        if (budget.remaining <= 0) return { status: 'capped' };
        budget.remaining--;
        bestG.set(newKey, g + 1);
        const idx = stateBoxes.length;
        stateBoxes.push(next);
        statePlayer.push(b);
        stateParent.push(s);
        statePushFrom.push(b);
        statePushTo.push(dest);
        statePushDir.push(k);
        stateG.push(g + 1);
        stateLow.push(newLow);
        stateKey.push(newKey);
        heapPush([g + 1 + hh, g + 1, seq++, idx]);
      }
    }
    for (const x of boxes) occupied[x] = 0;
  }
  return { status: 'dead' };
}

interface Room {
  squares: Uint8Array;
  boxes: number[];
  goals: number[];
  /** The room's door squares (room side), where the player walks out. */
  doors: number[];
}

/**
 * Split the board into rooms: cut the dead floor squares of the player's row (the
 * hallway), then flood-fill what is left. Each part holding boxes is a room.
 * Null when the board is no set of rooms (a box starts in the hallway, a goal lies
 * in the hallway, or a part's boxes and goals do not match).
 */
function splitRooms(board: Board, live: Uint8Array): Room[] | null {
  const { w, h, wall, goals, boxes, player } = board;
  const size = w * h;
  const row = Math.floor(player / w);
  const cut = new Uint8Array(size);
  for (let x = 0; x < w; x++) {
    const i = row * w + x;
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const comp = new Int32Array(size).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < size; s++) {
    if (wall[s] || cut[s] || comp[s] >= 0) continue;
    const squares: number[] = [s];
    comp[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      const p = squares[i]!;
      const col = p % w;
      for (const dir of DIRS) {
        if (dir.dx === -1 && col === 0) continue;
        if (dir.dx === 1 && col === w - 1) continue;
        const q = p + dir.dy * w + dir.dx;
        if (q < 0 || q >= size) continue;
        if (!wall[q] && !cut[q] && comp[q] < 0) {
          comp[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    const mask = new Uint8Array(size);
    for (const sq of squares) mask[sq] = 1;
    // The door squares: room squares touching the hallway (the cut squares).
    const doors: number[] = [];
    for (const sq of squares) {
      const col = sq % w;
      const neighbours = [sq - w, sq + w];
      if (col > 0) neighbours.push(sq - 1);
      if (col < w - 1) neighbours.push(sq + 1);
      if (neighbours.some((q) => q >= 0 && q < size && cut[q])) doors.push(sq);
    }
    rooms.push({ squares: mask, boxes: [], goals: [], doors });
  }
  for (const b of boxes) {
    const c = comp[b];
    if (c < 0) return null;
    rooms[c]!.boxes.push(b);
  }
  for (const g of goals) {
    const c = comp[g];
    if (c < 0) return null;
    rooms[c]!.goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** Solve one level, room by room, and return its moves ('' when it gives up). */
function solveLevel(board: Board): string {
  const { w, h, wall, goals, boxes, player } = board;
  const size = w * h;
  const fullDist = pushDistances(w, wall, goals);
  const live = new Uint8Array(size);
  for (let s = 0; s < size; s++) {
    for (let gi = 0; gi < goals.length; gi++) {
      if (fullDist[gi]![s]! >= 0) {
        live[s] = 1;
        break;
      }
    }
  }
  const rooms = splitRooms(board, live);
  const budget: Budget = { remaining: PER_INSTANCE_BUDGET };
  if (rooms === null) {
    const sub: SubLevel = {
      w,
      h,
      wall,
      goals: goals.slice(),
      boxes: boxes.slice(),
      player,
      roomMask: null,
      doors: null,
    };
    let res = searchPushes(sub, budget, true);
    if (res.status === 'capped') res = searchPushes(sub, budget, false);
    return res.status === 'solved' ? res.moves : '';
  }
  let moves = '';
  let curPlayer = player;
  const roomBoxes: number[][] = rooms.map((r) => r.boxes.slice().sort((a, b) => a - b));
  const roomDone: boolean[] = rooms.map(() => false);
  for (let ri = 0; ri < rooms.length; ri++) {
    const room = rooms[ri]!;
    const subWall = wall.slice();
    for (let oj = 0; oj < rooms.length; oj++) {
      if (oj === ri) continue;
      const ob = roomDone[oj] ? rooms[oj]!.goals : roomBoxes[oj]!;
      for (const b of ob) subWall[b] = 1;
    }
    const sub: SubLevel = {
      w,
      h,
      wall: subWall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: roomBoxes[ri]!.slice(),
      player: curPlayer,
      roomMask: room.squares,
      // Every room but the last must leave the player able to walk out its door.
      doors: ri < rooms.length - 1 ? room.doors : null,
    };
    let res = searchPushes(sub, budget, true);
    if (res.status === 'capped') res = searchPushes(sub, budget, false);
    if (res.status !== 'solved') return '';
    moves += res.moves;
    curPlayer = endPosition(w, curPlayer, res.moves);
    roomDone[ri] = true;
    roomBoxes[ri] = room.goals.slice().sort((a, b) => a - b);
  }
  return moves;
}

for (const id of readdirSync(IN_DIR).sort()) {
  const text = readFileSync(join(IN_DIR, id, 'level.txt'), 'utf8');
  let moves = '';
  try {
    moves = solveLevel(parseBoard(text));
  } catch {
    moves = '';
  }
  mkdirSync(join(OUT_DIR, id), { recursive: true });
  writeFileSync(join(OUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
