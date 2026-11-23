"""Short-horizon double-streamline references; no configuration-space teacher.

The chord check is an optional offline geometric diagnostic, not an AMB model.
A reference section may be ahead of the actual COG and is not teleported to it.
"""
from dataclasses import dataclass
import numpy as np
from cofl.runtime.temporal_field import ego_points, rotation


def pair_state(pair):
    pair = np.asarray(pair)
    d = pair[..., 0, :]-pair[..., 1, :]
    return pair.mean(-2), np.linalg.norm(d,axis=-1), np.arctan2(d[...,1],d[...,0])-np.pi/2


def angle_error(a,b):
    return np.arctan2(np.sin(a-b),np.cos(a-b))


@dataclass(frozen=True)
class EndpointFlowConfig:
    period_s: float = .2
    integration_dt_s: float = .02
    horizon_s: float = .6
    speed_m_s: float = .2
    max_endpoint_speed_m_s: float = .5
    min_width_m: float = .60
    max_width_rate_m_s: float = .25
    max_yaw_rate_rad_s: float = .8
    max_reference_jump_m: float = .02
    max_reference_yaw_jump_rad: float = np.deg2rad(5)
    max_lookahead_m: float = 1.2
    monotonic_shrink: bool = True
    half_fov_rad: float = np.pi/4

    def __post_init__(self):
        numbers = [self.period_s,self.integration_dt_s,self.horizon_s,self.speed_m_s,
                   self.max_endpoint_speed_m_s,self.min_width_m,self.max_width_rate_m_s,
                   self.max_yaw_rate_rad_s,self.max_reference_jump_m,self.max_reference_yaw_jump_rad,
                   self.max_lookahead_m,self.half_fov_rad]
        if not all(np.isfinite(numbers)) or min(numbers)<=0 or self.horizon_s<self.period_s:
            raise ValueError('Positive finite rates, horizon >= execution period required')


class EndpointFlowPlanner:
    def __init__(self, config=None, *, continuous=True):
        self.config = config or EndpointFlowConfig()
        self.continuous = continuous
        self.previous = None
        self.timestamp = None

    def reset(self):
        self.previous = self.timestamp = None

    def advance_pair(self, elapsed):
        index = elapsed/self.config.integration_dt_s
        lo = int(np.floor(index)); hi = lo+1
        if self.previous is None or lo >= len(self.previous):
            return None
        if hi >= len(self.previous):
            return self.previous[lo].copy()
        return self.previous[lo]*(hi-index)+self.previous[hi]*(index-lo)

    def _inside_sector(self, pair, position, yaw):
        p = ego_points(pair,position,yaw)
        return bool(np.all((p[:,0]>0)&(np.abs(np.arctan2(p[:,1],p[:,0]))<=self.config.half_fov_rad)
                           &(np.linalg.norm(p,axis=1)<=5.)))

    @staticmethod
    def _free(pair, free):
        # Chord only. This cannot certify the other arms/rotors of an actual AMB.
        samples = pair[0]+np.linspace(0,1,max(3,int(np.linalg.norm(pair[1]-pair[0])/.02)+1))[:,None]*(pair[1]-pair[0])
        return bool(np.all(free(samples)))

    def _perpendicular(self, pair, field):
        vector = field(pair.mean(0)[None])[0]
        chord = pair[0]-pair[1]
        norm = np.linalg.norm(vector)*np.linalg.norm(chord)
        if not np.isfinite(vector).all() or norm<1e-8:
            return None
        angle = np.rad2deg(np.arccos(np.clip(np.dot(vector,chord)/norm,-1,1)))
        return float(angle) if 85-1e-7<=angle<=95+1e-7 else None

    def _integrate(self, pair, field, free):
        cfg = self.config
        path = [pair.copy()]
        self.integration_stop='horizon_complete'
        for _ in range(int(round(cfg.horizon_s/cfg.integration_dt_s))):
            v = field(pair)
            if not np.isfinite(v).all():
                self.integration_stop='field_support_missing'
                break
            center_v = v.mean(0)
            speed = np.linalg.norm(center_v)
            if speed<1e-6:
                self.integration_stop='field_zero'
                break
            scale = min(cfg.speed_m_s/speed,
                        cfg.max_endpoint_speed_m_s/max(np.linalg.norm(v,axis=1).max(),1e-8))
            d = pair[0]-pair[1]; dv=v[0]-v[1]; width=np.linalg.norm(d)
            width_rate = np.dot(d,dv)/width
            if cfg.monotonic_shrink and width_rate>1e-6:
                self.integration_stop='flow_would_expand_span'
                break
            yaw_rate = (d[0]*dv[1]-d[1]*dv[0])/width**2
            scale = min(scale,cfg.max_width_rate_m_s/max(abs(width_rate),1e-8),
                        cfg.max_yaw_rate_rad_s/max(abs(yaw_rate),1e-8))
            candidate = pair+cfg.integration_dt_s*scale*v
            new_width = np.linalg.norm(candidate[0]-candidate[1])
            # Exact discrete checks supplement the derivative-based rate scaling.
            _,_,new_yaw=pair_state(candidate);_,_,old_yaw=pair_state(pair)
            if (new_width<cfg.min_width_m or abs(new_width-width)>cfg.max_width_rate_m_s*cfg.integration_dt_s+1e-6
                or abs(angle_error(new_yaw,old_yaw))>cfg.max_yaw_rate_rad_s*cfg.integration_dt_s+1e-6
                or (cfg.monotonic_shrink and new_width>width+1e-6) or not self._free(candidate,free)):
                self.integration_stop=('minimum_span' if new_width<cfg.min_width_m else 'discrete_rate_or_chord_constraint')
                break
            pair=candidate;path.append(pair.copy())
        return np.asarray(path)

    def _join(self, trajectory, elapsed, field, free):
        """Retain the old start and first velocity; smoothly correct the remaining prefix."""
        if self.previous is None or not self.continuous:
            return trajectory
        cfg=self.config
        result=trajectory.copy()
        for i in range(min(len(result),int(round(cfg.period_s/cfg.integration_dt_s))+1)):
            old=self.advance_pair(elapsed+i*cfg.integration_dt_s)
            if old is None:
                return None
            # Two fixed samples preserve the discrete start position AND first velocity.
            s=np.clip((i-1)/max(cfg.period_s/cfg.integration_dt_s-1,1),0,1)
            weight=6*s**5-15*s**4+10*s**3
            result[i]=old*(1-weight)+result[i]*weight
        center,width,yaw=pair_state(result)
        dt=cfg.integration_dt_s
        if (self._perpendicular(result[0],field) is None or width.min()<cfg.min_width_m
            or np.max(np.abs(np.diff(width)))>cfg.max_width_rate_m_s*dt+1e-6
            or np.max(np.abs(angle_error(yaw[1:],yaw[:-1])))>cfg.max_yaw_rate_rad_s*dt+1e-6
            or np.max(np.linalg.norm(np.diff(result,axis=0),axis=-1))>cfg.max_endpoint_speed_m_s*dt+1e-6
            or (cfg.monotonic_shrink and np.max(np.diff(width))>1e-6)
            or any(not self._free(pair,free) for pair in result)):
            return None
        return result

    def plan(self, position, yaw, width_m, timestamp, field, free):
        cfg = self.config
        if not np.isfinite(width_m) or width_m<cfg.min_width_m or not np.isfinite(timestamp):
            raise ValueError('Measured width must be finite and at least the configured minimum')
        if self.timestamp is not None and timestamp<=self.timestamp:
            raise ValueError('Timestamps must strictly increase; reset after pose/clock jumps')
        elapsed=timestamp-self.timestamp if self.timestamp is not None else 0.
        old = self.advance_pair(elapsed) if self.continuous and self.timestamp is not None else None
        if self.continuous and self.previous is not None and old is None:
            return dict(status='reference_expired',timestamp_s=float(timestamp),mode='A')
        proposals=[]
        if old is not None:
            center, w, _ = pair_state(old)
            proposals.append(center+(old-center)*(width_m/w))
        # Candidate sections are in front of the measured COG, not assumed actual endpoints.
        for fwd in np.arange(.25,cfg.max_lookahead_m+.001,.1):
            for left in np.arange(-.5,.501,.1):
                center = np.array([fwd,left])@rotation(yaw).T+position
                for delta in np.deg2rad(np.arange(-40,41,5)):
                    normal=np.array([-np.sin(yaw+delta),np.cos(yaw+delta)])
                    proposals.append(np.stack((center+width_m/2*normal,center-width_m/2*normal)))
        pairs=np.asarray(proposals)
        local_pairs=ego_points(pairs,position,yaw)
        valid=np.all((local_pairs[...,0]>0)&(np.abs(np.arctan2(local_pairs[...,1],local_pairs[...,0]))<=cfg.half_fov_rad)
                     &(np.linalg.norm(local_pairs,axis=-1)<=5.),axis=1)
        center,w,heading=pair_state(pairs)
        local=ego_points(center,position,yaw)
        cost=np.linalg.norm(local,axis=1)+.5*abs(local[:,1])+.1*abs(angle_error(heading,yaw))
        if old is not None:
            old_center,_,old_heading=pair_state(old)
            jump=np.linalg.norm(center-old_center,axis=1)
            yaw_jump=abs(angle_error(heading,old_heading))
            valid&=(jump<=cfg.max_reference_jump_m)&(yaw_jump<=cfg.max_reference_yaw_jump_rad)
            cost=jump+width_m*yaw_jump
        ids=np.flatnonzero(valid)
        candidates=[];failure_counts={}
        if len(ids):
            sample=pairs[ids,0,None]+np.linspace(0,1,max(3,int(width_m/.02)+1))[None,:,None]*(pairs[ids,1,None]-pairs[ids,0,None])
            clear=np.all(free(sample),axis=1)
            vectors=field(center[ids]);chord=pairs[ids,0]-pairs[ids,1]
            norm=np.linalg.norm(vectors,axis=1)*w[ids]
            angles=np.rad2deg(np.arccos(np.clip(np.sum(vectors*chord,axis=1)/np.maximum(norm,1e-12),-1,1)))
            for j in np.flatnonzero(clear&np.isfinite(angles)&(norm>1e-8)&(angles>=85)&(angles<=95)):
                k=ids[j];candidates.append((cost[k],pairs[k],float(angles[j])))
        for _,pair,angle in sorted(candidates,key=lambda a:a[0]):
            trajectory=self._integrate(pair,field,free)
            if (len(trajectory)-1)*cfg.integration_dt_s < cfg.period_s-1e-8:
                failure_counts[self.integration_stop]=failure_counts.get(self.integration_stop,0)+1
                continue
            trajectory=self._join(trajectory,elapsed,field,free)
            if trajectory is None or not self._inside_sector(trajectory[0],position,yaw):
                failure_counts['join_or_sector_constraint']=failure_counts.get('join_or_sector_constraint',0)+1
                continue
            center,w,heading=pair_state(trajectory)
            if abs(w[0]-width_m)>1e-5:
                continue
            angle=self._perpendicular(trajectory[0],field)
            center_jump = None if old is None else float(np.linalg.norm(center[0]-old.mean(0)))
            yaw_jump = None if old is None else float(abs(angle_error(heading[0],pair_state(old)[2])))
            old_next=self.advance_pair(elapsed+cfg.integration_dt_s) if old is not None else None
            velocity_jump=None if old_next is None else float(np.linalg.norm(
                (trajectory[1]-trajectory[0]-(old_next-old))/cfg.integration_dt_s,axis=1).max())
            self.previous,self.timestamp=trajectory,timestamp
            return dict(status='ok', timestamp_s=float(timestamp), initial_width_m=float(w[0]),
                perpendicular_deg=angle, endpoints_world=trajectory.tolist(), cog_world=center.tolist(),
                width_m=w.tolist(), yaw_rad=np.unwrap(heading).tolist(), integration_dt_s=cfg.integration_dt_s,
                join_position_error_m=center_jump,join_yaw_error_rad=yaw_jump,
                join_endpoint_velocity_error_m_s=velocity_jump,
                joining='retained start and first discrete velocity, quintic prefix correction' if old is not None else 'initial selection',
                reference_is_ahead_of_cog=True, mode='A' if self.continuous else 'B')
        # No implicit discontinuous reselection in A: caller must explicitly reset or stop.
        warm_angle=None
        if old is not None:
            v=field(old.mean(0)[None])[0];d=old[0]-old[1];norm=np.linalg.norm(v)*np.linalg.norm(d)
            if np.isfinite(v).all() and norm>1e-8:
                warm_angle=float(np.rad2deg(np.arccos(np.clip(np.dot(v,d)/norm,-1,1))))
        return dict(status='no_feasible_pair',timestamp_s=float(timestamp),
                    candidates_passing_selection=len(candidates), integration_failures=failure_counts,
                    warm_pair_inside_current_sector=None if old is None else self._inside_sector(old,position,yaw),
                    warm_pair_perpendicular_deg=warm_angle,mode='A' if self.continuous else 'B')
