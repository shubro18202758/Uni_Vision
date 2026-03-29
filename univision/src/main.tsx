import React from "react";
import ReactDOM from "react-dom/client";
import { ReactFlowProvider } from "reactflow";
import { AppShell } from "./app/AppShell";
import { initBlockRegistry } from "./lib/blockRegistry";
import "./styles.css";
import "reactflow/dist/style.css";

// Initialise the dynamic block registry (fetches from backend, falls
// back to static defaults) then mount the React tree.
initBlockRegistry().finally(() => {
  ReactDOM.createRoot(document.getElementById("root")!).render(
    <React.StrictMode>
      <ReactFlowProvider>
        <AppShell />
      </ReactFlowProvider>
    </React.StrictMode>,
  );
});
