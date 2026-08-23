import hashlib, math, random
from collections import defaultdict

NIVELES=[1,2,3,4,5]
PRESUPUESTO=40
E_REF=3.0

def _is_prime(n):
    if n<2: return False
    if n in (2,3,5,7): return True
    if n%2==0 or n%3==0: return False
    i=5
    while i*i<=n:
        if n%i==0 or n%(i+2)==0: return False
        i+=6
    return True
def _next_prime(n):
    if n<=2: return 2
    c=n|1
    while not _is_prime(c): c+=2
    return c
def derive_seeds(master,n=10):
    s=[]
    for i in range(n):
        d=hashlib.sha256(f"{master}:run_{i}".encode()).hexdigest()
        s.append(_next_prime(10007+int(d,16)%900000))
    return s

def rasch_p(h,nivel): return 1.0/(1.0+math.exp(-1.5*(h-nivel)))
PEAK_MODEL=0.5   # peak assumed by any agent that models the curve (misspecifiable)
def bell_curve(gap,peak=PEAK_MODEL): return 0.08*math.exp(-((gap-peak)**2)/0.5)

class StudyEnv:
    """v4: fatiga normalizada + histeresis + shift de regimen + examen en h_eff."""
    def __init__(self, h0, lr_mult=1.0, phi=0.0, lambda_olv=0.01,
                 hysteresis=False, e_crit=1.3, shift_mult=1.0, shift_at=20,
                 peak_gap=PEAK_MODEL, peak_new=None, peak_at=20):
        self.h=h0; self.h0=h0; self.lr_mult=lr_mult
        self.phi=phi; self.phi0=phi
        self.lambda_olv=lambda_olv
        self.hysteresis=hysteresis; self.e_crit=e_crit
        self.shift_mult=shift_mult; self.shift_at=shift_at
        self.peak_gap=peak_gap; self.peak_new=peak_new; self.peak_at=peak_at
        self.esfuerzo=0.0

    def tick(self,s):
        if self.shift_mult!=1.0 and s==self.shift_at:
            self.phi=self.phi0*self.shift_mult
        if self.peak_new is not None and s==self.peak_at:
            self.peak_gap=self.peak_new

    @property
    def esf_n(self): return self.esfuerzo/E_REF
    @property
    def h_eff(self): return max(0.5, self.h - self.phi*self.esf_n**1.5)
    @property
    def lr_eff(self):
        # la fatiga degrada la CAPACIDAD de aprender, no solo la dificultad aparente
        return self.lr_mult*max(0.15, 1.0 - 0.7*self.phi*self.esf_n**1.5)

    def expected_gain(self,nivel):
        gap=nivel-self.h_eff; p=rasch_p(self.h_eff,nivel)
        return bell_curve(gap,self.peak_gap)*(p*1.0+(1-p)*0.3)*self.lr_eff
    def in_zpd(self,nivel):
        g=[self.expected_gain(k) for k in NIVELES]; mx=max(g)
        return (self.expected_gain(nivel)>=0.5*mx) if mx>0 else False

    def exercise(self,nivel,rng):
        p=rasch_p(self.h_eff,nivel)
        correcto=rng.random()<p
        gap=max(0.0,nivel-self.h_eff)
        latencia=rng.lognormvariate(math.log(5)+0.6*gap,0.4)
        self.esfuerzo=0.95*self.esfuerzo+latencia/20.0
        gain=bell_curve(nivel-self.h_eff,self.peak_gap)*(1.0 if correcto else 0.3)*self.lr_eff
        self.h=min(6.0,self.h+gain)
        return {"correcto":correcto,"latencia":round(latencia,2),"gain":gain}

    def descansar(self):
        if self.hysteresis and self.esf_n>self.e_crit: self.esfuerzo*=0.93
        else: self.esfuerzo*=0.7
        self.h=max(0.5,self.h-self.lambda_olv)

class GCfg:
    def __init__(self,**kw):
        self.lam_p=kw.get("lam_p",0.15); self.lam_c=kw.get("lam_c",0.02)
        self.theta=kw.get("theta",0.7)
        self.alpha_p=kw.get("alpha_p",0.3); self.alpha_c=kw.get("alpha_c",0.3)
        self.w_cp=kw.get("w_cp",-0.05); self.w_pc=kw.get("w_pc",0.05)
        self.g_mode=kw.get("g_mode","learned")
        self.n_drives=kw.get("n_drives",2); self.coupling=kw.get("coupling",True)
        self.coupling_mode=kw.get("coupling_mode","receiver")  # receiver (Eq.1 / design intent) | sender (legacy bug)
        self.eta=kw.get("eta",0.3); self.beta=kw.get("beta",0.2)
        self.g_init_p=kw.get("g_init_p",0.3); self.g_init_c=kw.get("g_init_c",0.5)
        self.g_naive=kw.get("g_naive",0.5)
        self.action_mode=kw.get("action_mode","softmax")   # softmax | argmax
        self.rest_mode=kw.get("rest_mode","gated")  # gated | drive | none
        self.coef=kw.get("coef",(4.0,-3.0,-3.0,1.5,4.0,-3.0))  # up_p,up_c,stay_k,stay_b,down_c,down_p          # gated | drive

class GammaAgent:
    def __init__(self,cfg,profile):
        self.cfg=cfg; self.p=profile
        self.d_prog=0.5; self.d_conf=0.5
        self.g_prog={k:cfg.g_init_p for k in NIVELES}
        self.g_conf={k:cfg.g_init_c for k in NIVELES}
        self.acc_ema={}; self.nivel=1; self.frust=0.0
        self.consec_high=0; self.abandono=False; self.dropout_session=None
    def _pressure(self,d): return min(1.0,abs(d-self.cfg.theta)/0.8)
    def basal(self):
        m=self.acc_ema.get(self.nivel,0.5)
        self.d_prog=min(2.0,self.d_prog+self.cfg.lam_p*m)
        if self.cfg.n_drives>=2:
            self.d_conf=min(2.0,self.d_conf+self.cfg.lam_c)
            if self.cfg.coupling:
                pc=self._pressure(self.d_conf); pp=self._pressure(self.d_prog)
                if self.cfg.coupling_mode=="receiver":
                    # Eq.(1): modulated by the RECEIVING drive's own pressure
                    m_to_prog, m_to_conf = pp, pc
                else:
                    m_to_prog, m_to_conf = pc, pp     # legacy: sender's pressure
                if self.d_conf>self.cfg.theta+0.05:
                    self.d_prog=max(0.0,self.d_prog+self.cfg.w_cp*m_to_prog)
                if self.d_prog>self.cfg.theta+0.05:
                    self.d_conf=max(0.0,self.d_conf+self.cfg.w_pc*m_to_conf)
    def select(self,rng,env):
        th=self.cfg.theta
        dp=self.d_prog-th
        dc=(self.d_conf-th) if self.cfg.n_drives>=2 else 0.0
        fn=self.frust/max(self.p["umbral"],0.1)
        strain=max(0.0,fn-0.7)/0.3
        if self.cfg.n_drives>=2:
            if self.cfg.rest_mode=="gated":
                l_rest=2*max(0.0,dc)*strain+4*strain
            elif self.cfg.rest_mode=="none":
                l_rest=-1e9          # rest unavailable: isolates drives from the frustration gate
            else:
                l_rest=3.0*max(0.0,dc)+2.0*strain-1.2
            c=self.cfg.coef
            logits=[c[0]*dp+c[1]*dc, c[2]*(abs(dp)+abs(dc))+c[3], c[4]*dc+c[5]*dp, l_rest]
        else:
            logits=[4*dp,-3*abs(dp)+1.5,-4*dp,4*strain]
        acts=["subir","mantener","bajar","descansar"]
        if self.cfg.action_mode=="argmax":
            return acts[max(range(4),key=lambda i:logits[i])]
        mx=max(logits); exps=[math.exp(min(50,l-mx)) for l in logits]
        tot=sum(exps); r=rng.random()*tot; cum=0.0
        for i,e in enumerate(exps):
            cum+=e
            if r<cum: return acts[i]
        return "mantener"
    def study(self,env,outcome,nivel):
        c=outcome["correcto"]; eb=self.acc_ema.get(nivel,0.5)
        self.acc_ema[nivel]=(1-self.cfg.beta)*eb+self.cfg.beta*float(c)
        if self.cfg.g_mode=="learned":
            a=self.acc_ema[nivel]
            self.g_prog[nivel]=(1-self.cfg.eta)*self.g_prog[nivel]+self.cfg.eta*(4.0*a*(1.0-a))
            self.g_conf[nivel]=(1-self.cfg.eta)*self.g_conf[nivel]+self.cfg.eta*a
        elif self.cfg.g_mode=="oracle":
            self.g_prog[nivel]=min(1.0,env.expected_gain(nivel)/0.042)
            self.g_conf[nivel]=rasch_p(env.h_eff,nivel)
        if self.cfg.g_mode=="naive": gp=gc=self.cfg.g_naive
        else: gp=self.g_prog[nivel]; gc=self.g_conf[nivel]
        if c:
            self.d_prog=max(0.0,self.d_prog-self.cfg.alpha_p*gp)
            if self.cfg.n_drives>=2:
                self.d_conf=max(0.0,self.d_conf-self.cfg.alpha_c*gc)
            if outcome["latencia"]<15: self.frust=max(0.0,self.frust-0.35)
        else:
            if self.cfg.n_drives>=2: self.d_conf=min(2.0,self.d_conf+0.1)
            self.frust+=0.25
            if eb>=0.7: self.frust+=0.3

class ReglaAgent:
    def __init__(self,profile,rest=True):
        self.p=profile; self.rest=rest; self.nivel=1; self.frust=0.0
        self.consec_high=0; self.abandono=False; self.dropout_session=None
        self.streak_c=0; self.streak_e=0; self.acc_ema={}
    def select(self,rng,env):
        if self.rest and self.frust>0.6*self.p["umbral"]: return "descansar"
        if self.streak_e>=2: return "bajar"
        if self.streak_c>=3: return "subir"
        return "mantener"
    def study(self,env,outcome,nivel):
        if outcome["correcto"]: self.streak_c+=1; self.streak_e=0
        else: self.streak_e+=1; self.streak_c=0

class BayesianNoObjAgent:
    """State estimation WITHOUT an objective model: tracks h_hat from outcomes and
    simply matches the level to estimated ability. Isolates estimation from
    privileged knowledge of where the learning curve peaks."""
    def __init__(self,profile,rest=True):
        self.p=profile; self.rest=rest; self.nivel=1; self.h_hat=2.5
        self.frust=0.0; self.consec_high=0; self.abandono=False
        self.dropout_session=None; self.acc_ema={}
    def select(self,rng,env):
        if self.rest and self.frust>0.6*self.p["umbral"]: return "descansar"
        best=min(NIVELES,key=lambda k: abs(k-self.h_hat))
        if best>self.nivel: return "subir"
        if best<self.nivel: return "bajar"
        return "mantener"
    def study(self,env,outcome,nivel):
        p=rasch_p(self.h_hat,nivel)
        self.h_hat+=0.3*(float(outcome["correcto"])-p)
        self.h_hat=max(0.5,min(6.0,self.h_hat))

class BayesianAgent:
    def __init__(self,profile,rest=True):
        self.p=profile; self.rest=rest; self.nivel=1; self.h_hat=2.5
        self.frust=0.0; self.consec_high=0; self.abandono=False
        self.dropout_session=None; self.acc_ema={}
    def select(self,rng,env):
        if self.rest and self.frust>0.6*self.p["umbral"]: return "descansar"
        best,bg=self.nivel,-1
        for k in NIVELES:
            gap=k-self.h_hat; p=rasch_p(self.h_hat,k)
            g=bell_curve(gap)*(p+(1-p)*0.3)
            if g>bg: best,bg=k,g
        if best>self.nivel: return "subir"
        if best<self.nivel: return "bajar"
        return "mantener"
    def study(self,env,outcome,nivel):
        p=rasch_p(self.h_hat,nivel)
        self.h_hat+=0.3*(float(outcome["correcto"])-p)
        self.h_hat=max(0.5,min(6.0,self.h_hat))

class FixedAgent:
    def __init__(self,profile,mode):
        self.p=profile; self.mode=mode; self.nivel=1; self.frust=0.0
        self.consec_high=0; self.abandono=False; self.dropout_session=None
        self.acc_ema={}
    def select(self,rng,env):
        if self.mode=="always1": t=1
        elif self.mode=="always5": t=5
        elif self.mode=="random": t=rng.choice(NIVELES)
        else: t=max(NIVELES,key=lambda k:env.expected_gain(k))
        if t>self.nivel: return "subir"
        if t<self.nivel: return "bajar"
        return "mantener"
    def study(self,env,outcome,nivel): pass

def shared_frust_update(agent,outcome,eb):
    if outcome["correcto"]:
        if outcome["latencia"]<15: agent.frust=max(0.0,agent.frust-0.35)
    else:
        agent.frust+=0.25
        if eb>=0.7: agent.frust+=0.3

def gen_profiles(n=100,seed=2026):
    rng=random.Random(seed)
    return [{"h0":round(rng.uniform(1.5,3.5),2),
             "lr_mult":round(rng.uniform(0.7,1.3),2),
             "umbral":round(rng.uniform(1.8,3.0),1)} for _ in range(n)]

def run_episode(agent,profile,seed,is_gamma,env_kw,log=False):
    env=StudyEnv(profile["h0"],lr_mult=profile["lr_mult"],**env_kw)
    rng=random.Random(seed); traj=[]; zpd=0; nst=0; ndesc=0; s=0
    for s in range(PRESUPUESTO):
        if agent.abandono: break
        env.tick(s)
        if is_gamma: agent.basal()
        a=agent.select(rng,env)
        if a=="subir": agent.nivel=min(5,agent.nivel+1)
        elif a=="bajar": agent.nivel=max(1,agent.nivel-1)
        out=None; inz=False
        if a=="descansar":
            env.descansar(); agent.frust*=0.5; ndesc+=1
        else:
            inz=env.in_zpd(agent.nivel)
            out=env.exercise(agent.nivel,rng); nst+=1
            if inz: zpd+=1
            if is_gamma: agent.study(env,out,agent.nivel)
            else:
                eb=agent.acc_ema.get(agent.nivel,0.5)
                agent.acc_ema[agent.nivel]=0.8*eb+0.2*float(out["correcto"])
                shared_frust_update(agent,out,eb)
                agent.study(env,out,agent.nivel)
        if agent.frust>=0.95*profile["umbral"]: agent.consec_high+=1
        else: agent.consec_high=0
        if agent.consec_high>=3 and not agent.abandono:
            agent.abandono=True; agent.dropout_session=s
        if log:
            traj.append({"s":s,"nivel":agent.nivel,"action":a,
                "correcto":out["correcto"] if out else None,
                "d_prog":round(getattr(agent,"d_prog",0),3),
                "d_conf":round(getattr(agent,"d_conf",0),3),
                "frust":round(agent.frust,2),"h":round(env.h,3),
                "h_eff":round(env.h_eff,3),"esf_n":round(env.esf_n,3),"in_zpd":inz})
    m={"ganancia":round(env.h-profile["h0"],3),
       "h_final":round(env.h,3),
       "exam":round(rasch_p(env.h,3),3),
       "exam_taken":round(0.0 if agent.abandono else rasch_p(env.h_eff,3),3),
       "exam_eff":round(rasch_p(env.h_eff,3),3),
       "esf_n_final":round(env.esf_n,3),
       "tiz":round(zpd/max(1,nst),3),
       "tib_full":round(zpd/PRESUPUESTO,3),
       "dropout":agent.abandono,"dropout_session":agent.dropout_session,
       "n_descansar":ndesc,"n_study":nst,"nivel_final":agent.nivel,
       "sessions_used":s+1}
    if is_gamma and agent.cfg.g_mode in ("learned","oracle","fixed","naive"):
        gains=[env.expected_gain(k) for k in NIVELES]
        gps=[agent.g_prog[k] for k in NIVELES]
        mg=sum(gains)/5; mp=sum(gps)/5
        cov=sum((gains[i]-mg)*(gps[i]-mp) for i in range(5))
        vg=math.sqrt(sum((g-mg)**2 for g in gains)); vp=math.sqrt(sum((g-mp)**2 for g in gps))
        m["g_validity"]=round(cov/(vg*vp),3) if vg>0 and vp>0 else None
        m["g_prog_final"]={str(k):round(agent.g_prog[k],3) for k in NIVELES}
        m["true_gain_final"]={str(k):round(env.expected_gain(k),4) for k in NIVELES}
    return m,traj

CFG_BASE=dict(lam_p=0.15,alpha_p=0.3,alpha_c=0.3,theta=0.7)

def make_agent(brazo,profile):
    if brazo=="gamma_learned":  return GammaAgent(GCfg(**CFG_BASE,g_mode="learned"),profile),True
    if brazo=="gamma_learned_argmax": return GammaAgent(GCfg(**CFG_BASE,g_mode="learned",action_mode="argmax"),profile),True
    if brazo=="gamma_learned_restdrive": return GammaAgent(GCfg(**CFG_BASE,g_mode="learned",rest_mode="drive"),profile),True
    if brazo=="gamma_naive":    return GammaAgent(GCfg(**CFG_BASE,g_mode="naive"),profile),True
    if brazo=="gamma_no_learning": return GammaAgent(GCfg(**CFG_BASE,g_mode="fixed"),profile),True
    if brazo=="gamma_oracle":   return GammaAgent(GCfg(**CFG_BASE,g_mode="oracle"),profile),True
    if brazo=="regla":          return ReglaAgent(profile,rest=True),False
    if brazo=="regla_norest":   return ReglaAgent(profile,rest=False),False
    if brazo=="bayesian":       return BayesianAgent(profile,rest=True),False
    if brazo=="bayesian_norest":return BayesianAgent(profile,rest=False),False
    raise ValueError(brazo)
