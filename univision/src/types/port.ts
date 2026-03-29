/** Extensible port type — backend may register new types at runtime. */
export type PortType = string;

export type PortDirection = "input" | "output";

export interface PortDefinition {
  id: string;
  name: string;
  type: PortType;
  direction: PortDirection;
}
