import { useCallback, useEffect, useState } from 'react';
import type { TopLevelRoute } from '../types/domain';

const staticRoutes = new Set<TopLevelRoute>(['final', 'audit']);
const caseRoutePattern = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;

export function routeFromLocation(): TopLevelRoute {
  const parts = window.location.pathname.split('/').filter(Boolean);
  const last = parts[parts.length - 1] as TopLevelRoute | undefined;
  if (!last || last === 'app' || last === 'index.html') return 'day';
  if (staticRoutes.has(last) || caseRoutePattern.test(last)) return last;
  return 'day';
}

export function useRouteState(): [TopLevelRoute, (route: TopLevelRoute) => void] {
  const [route, setRouteState] = useState<TopLevelRoute>(routeFromLocation);

  const setRoute = useCallback((nextRoute: TopLevelRoute) => {
    const prefix = window.location.pathname.includes('/app/') ? '/app' : '';
    window.history.pushState({}, '', `${prefix}/${nextRoute}${window.location.search}${window.location.hash}`);
    setRouteState(nextRoute);
  }, []);

  useEffect(() => {
    const onPop = () => setRouteState(routeFromLocation());
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);

  return [route, setRoute];
}
