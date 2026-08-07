import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { RefObject } from 'react';
import type { Case, MediaVersion } from '../types/api';
import { caseToSourceFrame, clamp, formatTime, frameToSeconds, secondsToFrame } from '../utils/frame';

export interface VideoController {
  videoRef: RefObject<HTMLVideoElement>;
  wrapperRef: RefObject<HTMLDivElement>;
  caseFrame: number;
  sourceFrame: number;
  timeText: string;
  playing: boolean;
  mediaError?: string;
  seekToFrame: (frame: number) => void;
  seekToSourceFrame: (sourceFrame: number) => void;
  stepFrame: (delta: number) => void;
  togglePlay: () => void;
  toggleFullscreen: () => void;
  retryMedia: () => void;
  setMediaError: (message?: string) => void;
}

export function useVideoController(
  manifest: Case,
  version: MediaVersion | undefined,
  caseFrame: number,
  setCaseFrame: (frame: number) => void,
): VideoController {
  const videoRef = useRef<HTMLVideoElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [playing, setPlaying] = useState(false);
  const [mediaError, setMediaError] = useState<string>();
  const lastVersionPath = useRef<string | undefined>(version?.path);

  const maxFrame = Math.max(0, manifest.frame_count - 1);
  const seekToFrame = useCallback(
    (frame: number) => {
      const nextFrame = clamp(Math.round(frame), 0, maxFrame);
      setCaseFrame(nextFrame);
      const video = videoRef.current;
      if (video && Number.isFinite(video.duration || 0)) {
        video.currentTime = frameToSeconds(manifest, nextFrame);
      }
    },
    [manifest, maxFrame, setCaseFrame],
  );

  const seekToSourceFrame = useCallback(
    (sourceFrame: number) => {
      seekToFrame(sourceFrame - manifest.source_frame_start);
    },
    [manifest.source_frame_start, seekToFrame],
  );

  const stepFrame = useCallback(
    (delta: number) => {
      const video = videoRef.current;
      if (video && !video.paused) video.pause();
      seekToFrame(caseFrame + delta);
    },
    [caseFrame, seekToFrame],
  );

  const togglePlay = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      void video.play().catch((err) => setMediaError(err instanceof Error ? err.message : String(err)));
    } else {
      video.pause();
    }
  }, []);

  const toggleFullscreen = useCallback(() => {
    const wrapper = wrapperRef.current;
    if (!wrapper) return;
    if (document.fullscreenElement) {
      void document.exitFullscreen();
    } else {
      void wrapper.requestFullscreen();
    }
  }, []);

  const retryMedia = useCallback(() => {
    setMediaError(undefined);
    videoRef.current?.load();
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const onTimeUpdate = () => setCaseFrame(secondsToFrame(manifest, video.currentTime));
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    const onError = () => setMediaError('Media could not be loaded. Check package media files and Range support.');
    video.addEventListener('timeupdate', onTimeUpdate);
    video.addEventListener('play', onPlay);
    video.addEventListener('pause', onPause);
    video.addEventListener('error', onError);
    video.disablePictureInPicture = true;
    return () => {
      video.removeEventListener('timeupdate', onTimeUpdate);
      video.removeEventListener('play', onPlay);
      video.removeEventListener('pause', onPause);
      video.removeEventListener('error', onError);
    };
  }, [manifest, setCaseFrame]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    if (lastVersionPath.current !== version?.path) {
      lastVersionPath.current = version?.path;
      video.currentTime = frameToSeconds(manifest, caseFrame);
    }
  }, [caseFrame, manifest, version?.path]);

  return useMemo(
    () => ({
      videoRef,
      wrapperRef,
      caseFrame,
      sourceFrame: caseToSourceFrame(manifest, caseFrame),
      timeText: formatTime(frameToSeconds(manifest, caseFrame)),
      playing,
      mediaError,
      seekToFrame,
      seekToSourceFrame,
      stepFrame,
      togglePlay,
      toggleFullscreen,
      retryMedia,
      setMediaError,
    }),
    [
      caseFrame,
      manifest,
      mediaError,
      playing,
      retryMedia,
      seekToFrame,
      seekToSourceFrame,
      stepFrame,
      toggleFullscreen,
      togglePlay,
    ],
  );
}
