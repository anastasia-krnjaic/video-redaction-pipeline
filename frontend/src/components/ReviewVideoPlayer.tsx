import { Maximize2, Pause, Play, RotateCcw, StepBack, StepForward } from 'lucide-react';
import type { Case, MediaVersion } from '../types/api';
import type { VideoController } from '../hooks/useVideoController';
import { VersionSelector } from './VersionSelector';

interface Props {
  manifest: Case;
  selectedRole: string;
  roles: MediaVersion[];
  finalRole?: MediaVersion;
  version?: MediaVersion;
  controller: VideoController;
  onVersionChange: (role: string) => void;
}

export function ReviewVideoPlayer({
  controller,
  finalRole,
  manifest,
  onVersionChange,
  roles,
  selectedRole,
  version,
}: Props) {
  const source = version?.path || manifest.source_path;
  return (
    <section className="player-column" aria-label={`${manifest.display_name} video player`}>
      <div className="version-row">
        <VersionSelector
          finalRole={finalRole}
          roles={roles}
          selectedRole={selectedRole}
          onChange={onVersionChange}
        />
        <div className="source-range">
          Source {manifest.source_frame_start}-{manifest.source_frame_end}
        </div>
      </div>
      <div className="player-wrapper" ref={controller.wrapperRef}>
        <div className="video-stage">
          <video
            ref={controller.videoRef}
            data-case={manifest.case_id}
            disablePictureInPicture
            playsInline
            preload="metadata"
            src={`/${source}`}
            onClick={controller.togglePlay}
          />
          <div className="frame-hud">
            Source frame {controller.sourceFrame} · Case frame {controller.caseFrame} · {controller.timeText}
          </div>
          {controller.mediaError ? (
            <div className="media-error visible">
              <span>{controller.mediaError}</span>
              <button type="button" onClick={controller.retryMedia}>
                <RotateCcw size={14} aria-hidden="true" /> Retry
              </button>
            </div>
          ) : null}
        </div>
        <div className="player-controls">
          <div className="controls-top">
            <button type="button" onClick={controller.togglePlay} title="Play or pause">
              {controller.playing ? <Pause size={16} aria-hidden="true" /> : <Play size={16} aria-hidden="true" />}
              <span>{controller.playing ? 'Pause' : 'Play'}</span>
            </button>
            <div className="time-readout">{controller.timeText}</div>
            <button type="button" onClick={() => controller.stepFrame(-1)} title="Step back one frame">
              <StepBack size={16} aria-hidden="true" />
            </button>
            <button type="button" onClick={() => controller.stepFrame(1)} title="Step forward one frame">
              <StepForward size={16} aria-hidden="true" />
            </button>
            <button type="button" onClick={controller.toggleFullscreen} title="Fullscreen">
              <Maximize2 size={16} aria-hidden="true" />
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}
