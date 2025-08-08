library ieee;
use     ieee.std_logic_1164.all;

entity and3 is
  port (
    a : in  std_logic;   -- e.g. TVALID
    b : in  std_logic;   --      TREADY
    c : in  std_logic;   --      TLAST
    y : out std_logic    -- = a and b and c  (one-cycle pulse)
  );
end entity and3;

architecture rtl of and3 is
begin
  y <= a and b and c;
end architecture rtl;