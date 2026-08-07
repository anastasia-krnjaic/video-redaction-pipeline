import { useCallback, useEffect, useState } from 'react';
import type { TopLevelRoute } from '../types/domain';

const routes = new Set<TopLevelRoute>(['day', 'night', 'final', 'audit']);

export function routeFromLocation(): TopLevelRoute {
  const parts = window.location.pathname.split('/').filter(Boolean);
  const last = parts[parts.length - 1] as TopLevelRoute | undefined;
  return last && routes.has(last) ? last : 'day';
}

export function useRouteState(): [TopLevelRoute, (route: TopLevelRoute) => void] {
  const [route, setRouteState] = useState<TopLevelRoute>(routeFromLocation);

  const setRoute = useCallback((nextRoute: TopLevelRoute) => {
    const prefix = window.location.pathname.includes('/app/') ? '/app' : '';
    window.history.pushState({}, '', `${prefix}/${nextRoute}`);
    setRouteState(nextRoute);
  }, []);

  useEffect(() => {
    const onPop = () => setRouteState(routeFromLocation());
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);

  return [route, setRoute];
}
