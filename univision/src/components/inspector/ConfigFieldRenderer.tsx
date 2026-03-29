import type { ConfigFieldSchema } from "../../types/configSchema";
import { NumberField } from "./fields/NumberField";
import { SelectField } from "./fields/SelectField";
import { TextField } from "./fields/TextField";
import { TextareaField } from "./fields/TextareaField";
import { ToggleField } from "./fields/ToggleField";

interface Props {
  field: ConfigFieldSchema;
  value: string | number | boolean | undefined;
  onChange: (value: string | number | boolean) => void;
}

export function ConfigFieldRenderer({ field, value, onChange }: Props) {
  if (field.type === "number") {
    return <NumberField label={field.label} onChange={onChange} value={value} />;
  }

  if (field.type === "select") {
    return <SelectField label={field.label} onChange={onChange} options={field.options ?? []} value={value} />;
  }

  if (field.type === "toggle") {
    return <ToggleField label={field.label} onChange={onChange} value={value} />;
  }

  if (field.type === "textarea") {
    return <TextareaField label={field.label} onChange={onChange} value={value} placeholder={field.placeholder} />;
  }

  return <TextField label={field.label} onChange={onChange} value={value} />;
}
