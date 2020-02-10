(
s = Server.local.boot;
s.waitForBoot({
TempoClock.default.tempo = 120/60;


// A slightly more flexible example using arguments8// and a self−terminating envelope (doneAction: 2)
SynthDef("mySine2", {
	arg freq = 440, amp = 0.8, pan=0;
	var env, sig;
	env = Env.perc(level: amp).kr(2);
	env = Env.adsr(0.05,0.1,0.9,3);
	varsnd = SinOsc.ar(freq, 0, env);
	Out.ar(0, snd);13}).add;





//- make a synth
SynthDef(\tone, {
	arg freq=40, nharm=12, detune=0.2, gate=0,
	pan=0, amp=1, out=0;
	var sig, env;
	env = EnvGen.kr(Env.adsr(0.05,0.1,0.9,3), gate);
	sig = Blip.ar(
		freq *
		LFNoise1.kr(0.2!16).bipolar(detune.neg, detune).midiratio,
		nharm
	);
	sig = sig * LFNoise1.kr(0.5!16).exprange(0.1, 1);
	sig = Splay.ar(sig);
	sig = Balance2.ar(sig[0], sig[1], pan);
	sig = sig * env * amp;
	Out.ar(out, sig);
}).store;


//- oscs
OSCdef.new(
	\noteOn,
	{
		arg msg, time, addr, port;
		//[msg, time, addr, port].postln;
		msg.postln;
		x.set(\gate, msg[1]);
	},
	'/ctrl/on'
);
OSCdef.new(
	\noteOff,
	{
		arg msg, time, addr, port;
		//[msg, time, addr, port].postln;
		msg.postln;
		x.set(\gate, msg[1]);
	},
	'/ctrl/off'
);

OSCdef.new(
	\rot_a,
	{
		arg msg, time, addr, port;
		//[msg, time, addr, port].postln;
		var n_freq = msg[1].linexp(-64,64,21,2000);
		x.set(\freq, n_freq);
		msg.post; "->".post; n_freq.postln;
	},
	'/ctrl/a'
);
OSCdef.new(
	\rot_b,
	{
		arg msg, time, addr, port;
		//[msg, time, addr, port].postln;
		var n_nharm = msg[1].linlin(-64,64,1,36);
		x.set(\nharm, n_nharm);
		msg.post; "->".post; n_nharm.postln;
	},
	'/ctrl/b'
);
OSCdef.new(
	\rot_c,
	{
		arg msg, time, addr, port;
		//[msg, time, addr, port].postln;
		var n_det = msg[1].linexp(-64,64,1,36);
		x.set(\detune, n_det);
		msg.post; "->".post; n_det.postln;
	},
	'/ctrl/c'
);
OSCdef.new(
	\rot_d,
	{
		arg msg, time, addr, port;
		//[msg, time, addr, port].postln;
		var n_pan = msg[1].linlin(-32,32,-1,1);
		x.set(\pan, n_pan);
		msg.post; "->".post; n_pan.postln;
	},
	'/ctrl/d'
);

// start running
x = Synth(\tone, [\gate, 0]);


});
)

