library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

------------------------------------------------------------------------
-- cycle_counter_wrapper.vhd
--
-- Monitors two sets of 3 input signals ("start" and "done"),
-- AND-gates each triple to produce a one-cycle pulse,
-- and uses a cycle_counter to measure the interval.
------------------------------------------------------------------------
entity cycle_counter_wrapper is
  port (
    -- Clock & reset
    clk           : in  std_logic;
    rst_n         : in  std_logic;

    -- "Start" signals (hook these to your master TVALID/TREADY/TLAST)
    start_valid   : in  std_logic;
    start_ready   : in  std_logic;
    start_last    : in  std_logic;

    -- "Done"  signals (hook these to your slave  TVALID/TREADY/TLAST)
    done_valid    : in  std_logic;
    done_ready    : in  std_logic;
    done_last     : in  std_logic;

    -- 32-bit cycle count result
    cycles_o      : out std_logic_vector(31 downto 0)
  );
end entity cycle_counter_wrapper;


architecture rtl of cycle_counter_wrapper is

  -- Internal one-cycle pulses
  signal start_pulse : std_logic;
  signal done_pulse  : std_logic;

  -- 3-input AND cell
  component and3 is
    port (
      a : in  std_logic;
      b : in  std_logic;
      c : in  std_logic;
      y : out std_logic
    );
  end component and3;

  -- Cycle counter
  component cycle_counter is
    port (
      clk      : in  std_logic;
      rst_n    : in  std_logic;
      start_i  : in  std_logic;
      done_i   : in  std_logic;
      cycles_o : out std_logic_vector(31 downto 0)
    );
  end component cycle_counter;

begin

  -- Generate a one-cycle "start" when all three start_* are high
  start_and3 : and3
    port map (
      a => start_valid,
      b => start_ready,
      c => start_last,
      y => start_pulse
    );

  -- Generate a one-cycle "done" when all three done_* are high
  done_and3 : and3
    port map (
      a => done_valid,
      b => done_ready,
      c => done_last,
      y => done_pulse
    );

  -- Count cycles between start_pulse and done_pulse
  counter_inst : cycle_counter
    port map (
      clk      => clk,
      rst_n    => rst_n,
      start_i  => start_pulse,
      done_i   => done_pulse,
      cycles_o => cycles_o
    );

end architecture rtl;
