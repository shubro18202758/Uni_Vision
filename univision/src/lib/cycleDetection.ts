import type { GraphBlock } from "../types/block";
import type { GraphConnection } from "../types/connection";

export function hasCycle(blocks: GraphBlock[], connections: GraphConnection[]) {
  const indegree = new Map<string, number>();
  const adjacency = new Map<string, string[]>();

  blocks.forEach((block) => {
    indegree.set(block.id, 0);
    adjacency.set(block.id, []);
  });

  connections.forEach((connection) => {
    adjacency.get(connection.source)?.push(connection.target);
    indegree.set(connection.target, (indegree.get(connection.target) ?? 0) + 1);
  });

  const queue = [...indegree.entries()].filter(([, count]) => count === 0).map(([id]) => id);
  let visited = 0;

  while (queue.length > 0) {
    const current = queue.shift()!;
    visited += 1;

    for (const next of adjacency.get(current) ?? []) {
      const nextCount = (indegree.get(next) ?? 0) - 1;
      indegree.set(next, nextCount);
      if (nextCount === 0) {
        queue.push(next);
      }
    }
  }

  return visited !== blocks.length;
}
