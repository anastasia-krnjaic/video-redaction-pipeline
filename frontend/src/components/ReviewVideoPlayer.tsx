import { RotateCcw } from 'lucide-react';
import type { ReactNode } from 'react';
import type { Case, MediaVersion } from '../types/api';
import type { VideoController } from '../hooks/useVideoController';
import type { TaskWithDecision } from '../types/domain';
import { PlayerChrome } from './PlayerChrome';
import { VersionSelector } from './VersionSelector';

interface Props {
  children?: ReactNode;
  manifest: Case;
  selectedRole: string;
  roles: MediaVersion[];
  finalRole?: MediaVersion;
  version?: MediaVersion;
  controller: VideoController;
  selectedTaskId?: string | null;
  tasks?: TaskWithDecision[];
  onVersionChange: (role: string) => void;
  onSelectTask?: (task: TaskWithDecision) => void;
}

export function ReviewVideoPlayer({
  children,
  controller,
  finalRole,
  manifest,
  onSelectTask,
  onVersionChange,
  roles,
  selectedRole,
  selectedTaskId,
  tasks,
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
            data-media-path={source}
            disablePictureInPicture
            playsInline
            preload="metadata"
            src={`/${source}`}
            onClick={controller.togglePlay}
          />
          {children}
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
        <PlayerChrome
          controller={controller}
          manifest={manifest}
          selectedTaskId={selectedTaskId}
          tasks={tasks}
          onSelectTask={onSelectTask}
        />
      </div>
    </section>
  );
}
