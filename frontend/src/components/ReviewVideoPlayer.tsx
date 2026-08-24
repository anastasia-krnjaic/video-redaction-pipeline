import { Download, RotateCcw } from 'lucide-react';
import type { ReactNode } from 'react';
import type { Case, MediaVersion } from '../types/api';
import type { VideoController } from '../hooks/useVideoController';
import type { ReviewGroup } from '../types/domain';
import { PlayerChrome } from './PlayerChrome';
import { VersionSelector, roleLabel } from './VersionSelector';

interface Props {
  children?: ReactNode;
  manifest: Case;
  selectedRole: string;
  sessionId?: string;
  roles: MediaVersion[];
  finalRole?: MediaVersion;
  version?: MediaVersion;
  controller: VideoController;
  selectedTaskId?: string | null;
  tasks?: ReviewGroup[];
  onVersionChange: (role: string) => void;
  onSelectTask?: (task: ReviewGroup) => void;
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
  sessionId,
  selectedTaskId,
  tasks,
  version,
}: Props) {
  const source = version?.path || manifest.source_path;
  const downloadOptions = downloadMenuOptions(manifest, roles, sessionId, finalRole);
  const finalDownloadAvailable = downloadOptions.some((option) => option.role === 'final_reviewed' && option.available);
  return (
    <section className="player-column" aria-label={`${manifest.display_name} video player`}>
      <div className="player-wrapper" ref={controller.wrapperRef}>
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
        <div className="download-row" aria-label={`${manifest.display_name} downloads`}>
          {downloadOptions.map((option) => (
            option.available ? (
              <a key={option.role} className="download-pill available" href={option.url} title={option.status}>
                <Download size={14} aria-hidden="true" /> {option.label}
              </a>
            ) : (
              <span key={option.role} className="download-pill unavailable" aria-disabled="true" title={option.status}>
                {option.label}
              </span>
            )
          ))}
        </div>
        <div className="download-secondary-row">
          <a className="download-link package-link" href={deliveryPackageUrl(manifest.case_id, sessionId)}>
            <Download size={14} aria-hidden="true" /> Review package manifest
          </a>
          {finalDownloadAvailable ? (
            <a className="download-link package-link" href={deliveryPackageZipUrl(manifest.case_id, sessionId)}>
              <Download size={14} aria-hidden="true" /> Final package .zip
            </a>
          ) : (
            <span className="download-package-status unavailable">Final package not available</span>
          )}
        </div>
      </div>
    </section>
  );
}

interface DownloadOption {
  available: boolean;
  label: string;
  role: 'original' | 'automatic' | 'latest_reviewed' | 'final_reviewed';
  status: string;
  url: string;
}

const downloadOrder: DownloadOption['role'][] = ['original', 'automatic', 'latest_reviewed', 'final_reviewed'];

function downloadMenuOptions(
  manifest: Case,
  roles: MediaVersion[],
  sessionId?: string,
  finalRole?: MediaVersion,
): DownloadOption[] {
  const byRole = new Map(roles.map((role) => [role.role, role]));
  if (finalRole) byRole.set('final_reviewed', finalRole);
  return downloadOrder.map((role) => {
    const version = byRole.get(role);
    const label = role === 'final_reviewed' ? 'Final' : version ? roleLabel(version) : unavailableLabel(role);
    return {
      available: Boolean(version),
      label: role === 'latest_reviewed' && version ? 'Latest reviewed (current)' : label,
      role,
      status: downloadStatus(role, Boolean(version)),
      url: downloadUrl(manifest.case_id, role, sessionId),
    };
  });
}

function unavailableLabel(role: DownloadOption['role']): string {
  if (role === 'original') return 'Original';
  if (role === 'automatic') return 'Automatic';
  if (role === 'latest_reviewed') return 'Latest reviewed';
  return 'Final';
}

function downloadStatus(role: DownloadOption['role'], available: boolean): string {
  if (!available) {
    if (role === 'latest_reviewed') return 'No current Latest reviewed pointer';
    if (role === 'final_reviewed') return 'No current Final';
    return 'Missing from case manifest';
  }
  if (role === 'latest_reviewed') return 'Current authoritative media';
  if (role === 'final_reviewed') return 'Current final media';
  return 'Available';
}

function downloadUrl(caseId: string, role: DownloadOption['role'], sessionId?: string): string {
  const suffix = sessionId ? `&session_id=${encodeURIComponent(sessionId)}` : '';
  return `/api/v1/cases/${encodeURIComponent(caseId)}/download?role=${encodeURIComponent(role)}${suffix}`;
}

function deliveryPackageUrl(caseId: string, sessionId?: string): string {
  const suffix = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
  return `/api/v1/cases/${encodeURIComponent(caseId)}/delivery-package${suffix}`;
}

function deliveryPackageZipUrl(caseId: string, sessionId?: string): string {
  const suffix = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
  return `/api/v1/cases/${encodeURIComponent(caseId)}/delivery-package.zip${suffix}`;
}
