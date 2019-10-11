# Energy Detection and Signal Matching in SETI

## Background

(Skipping generic text about how important SETI is)

A lot of the data that Breakthrough Listen is collecting is in the form of dynamic spectrums, which we call waterfalls. They look like this:

![Example Waterfall](./imgs/waterfall_example.png)

The x axis is Frequency, the y axis is time, and each point has a power reading representing how much energy is coming from that frequency at that timestep.

If you look at the part where the frequency is about 1060 Mhz, you can see a thin bright line in the plot, that's a signal, and if you look closely, there are many more bright lines in the plot, which are of course all signals picked up by the telescope.

The goal of this project is building a database of "seen" signals and constructing a pipeline for picking out signals from every observation and using dimensionality reduction to match them to seen ones. If we find a signal that's unseen before from any of our observations, that means we've found ET! (or a bug)
