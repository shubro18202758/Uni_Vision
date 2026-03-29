import { getCategoryColor } from "../../constants/categories";

export function BlockCategoryBar({ category }: { category: string }) {
  return <div className="h-1 rounded-none" style={{ backgroundColor: getCategoryColor(category) }} />;
}
