import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const [inputPath, outputPath, sheetName, range = "A1:Z250"] = process.argv.slice(2);
if (!inputPath || !outputPath) {
  throw new Error(
    "usage: node inspect_workbook.mjs <input.xlsx> <output.json> [sheet] [range]",
  );
}

const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const sheetSummary = await workbook.inspect({
  kind: "sheet",
  include: "id,name",
  maxChars: 20_000,
});
const result = {
  input_path: inputPath,
  sheet_summary_ndjson: sheetSummary.ndjson,
};
if (sheetName) {
  const sheet = workbook.worksheets.getItem(sheetName);
  result.selected_sheet = sheetName;
  result.selected_range = range;
  result.values = sheet.getRange(range).values;
}
await fs.writeFile(outputPath, `${JSON.stringify(result, null, 2)}\n`, "utf8");
