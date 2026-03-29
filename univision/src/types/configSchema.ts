/** Extensible config field type — backend may register new types. */
export type ConfigFieldType = string;

export interface ConfigFieldOption {
  label: string;
  value: string;
}

export interface ConfigFieldSchema {
  key: string;
  label: string;
  type: ConfigFieldType;
  placeholder?: string;
  required?: boolean;
  min?: number;
  max?: number;
  options?: ConfigFieldOption[];
}
