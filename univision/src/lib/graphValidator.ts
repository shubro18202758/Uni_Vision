import { getBlockDefinition } from "./blockRegistry";
import { hasCycle } from "./cycleDetection";
import type { GraphBlock } from "../types/block";
import type { GraphConnection } from "../types/connection";
import type { ValidationIssue } from "../types/graph";

export function validateGraph(blocks: GraphBlock[], connections: GraphConnection[]): ValidationIssue[] {
  const issues: ValidationIssue[] = [];

  for (const connection of connections) {
    if (connection.source === connection.target) {
      issues.push({
        id: `self-${connection.id}`,
        level: "error",
        message: "A block cannot connect to itself.",
        connectionId: connection.id,
      });
    }

    const sourceBlock = blocks.find((block) => block.id === connection.source);
    const targetBlock = blocks.find((block) => block.id === connection.target);
    const sourceDef = sourceBlock ? getBlockDefinition(sourceBlock.type) : undefined;
    const targetDef = targetBlock ? getBlockDefinition(targetBlock.type) : undefined;
    const sourcePort = sourceDef?.outputs.find((port) => port.id === connection.sourceHandle);
    const targetPort = targetDef?.inputs.find((port) => port.id === connection.targetHandle);

    if (sourcePort && targetPort && sourcePort.type !== targetPort.type) {
      issues.push({
        id: `type-${connection.id}`,
        level: "error",
        message: `Port type mismatch: ${sourcePort.type} cannot connect to ${targetPort.type}.`,
        connectionId: connection.id,
      });
    }
  }

  const incomingByHandle = new Map<string, number>();
  connections.forEach((connection) => {
    const key = `${connection.target}:${connection.targetHandle}`;
    incomingByHandle.set(key, (incomingByHandle.get(key) ?? 0) + 1);
  });

  for (const [key, count] of incomingByHandle.entries()) {
    if (count > 1) {
      issues.push({
        id: `fanin-${key}`,
        level: "error",
        message: "Each input port accepts only one incoming connection.",
      });
    }
  }

  blocks.forEach((block) => {
    const definition = getBlockDefinition(block.type);
    if (!definition) {
      return;
    }

    definition.configSchema.forEach((field) => {
      if (field.required && (block.config[field.key] === "" || block.config[field.key] === undefined)) {
        issues.push({
          id: `config-${block.id}-${field.key}`,
          level: "error",
          message: `${block.label}: ${field.label} is required.`,
          blockId: block.id,
          portId: field.key,
        });
      }
    });
  });

  if (hasCycle(blocks, connections)) {
    issues.push({
      id: "cycle",
      level: "error",
      message: "The graph contains a cycle. Pipelines must stay acyclic.",
    });
  }

  return issues;
}
