// @ts-nocheck
import React from 'react';
import { ApplyPluginsType } from 'E:/0_Projects/基于集成学习的恶意PE软件特征检测与识别/mal_ana/frontend/node_modules/@umijs/runtime';
import * as umiExports from './umiExports';
import { plugin } from './plugin';

export function getRoutes() {
  const routes = [
  {
    "path": "/",
    "component": require('E:/0_Projects/基于集成学习的恶意PE软件特征检测与识别/mal_ana/frontend/src/.umi/plugin-layout/Layout.tsx').default,
    "routes": [
      {
        "path": "/",
        "component": require('@/pages/index').default,
        "exact": true
      },
      {
        "path": "/dashboard",
        "component": require('@/pages/dashboard/index').default,
        "exact": true
      },
      {
        "path": "/samples",
        "component": require('@/pages/samples/index').default,
        "exact": true
      },
      {
        "path": "/samples/:id",
        "component": require('@/pages/samples/detail').default,
        "exact": true
      },
      {
        "path": "/upload",
        "component": require('@/pages/upload/index').default,
        "exact": true
      }
    ]
  }
];

  // allow user to extend routes
  plugin.applyPlugins({
    key: 'patchRoutes',
    type: ApplyPluginsType.event,
    args: { routes },
  });

  return routes;
}
