/// <reference types="vite/client" />

declare module 'react-plotly.js' {
  import * as React from 'react';
  import { PlotParams } from 'react-plotly.js';
  const Plot: React.ComponentType<any>;
  export default Plot;
}
