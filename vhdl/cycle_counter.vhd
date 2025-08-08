library ieee; 
use ieee.std_logic_1164.all; 
use ieee.numeric_std.all;

entity cycle_counter is
  port (
    clk      : in  std_logic;
    rst_n    : in  std_logic;
    start_i  : in  std_logic;
    done_i   : in  std_logic;
    cycles_o : out std_logic_vector(31 downto 0)
  );
end cycle_counter;

architecture rtl of cycle_counter is
  signal run   : std_logic := '0';
  signal count : unsigned(31 downto 0) := (others => '0');
begin
  process(clk)
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        run   <= '0';
        count <= (others=>'0');
      else
        if start_i = '1' then
          run   <= '1';
          count <= (others=>'0');
        elsif done_i = '1' then
          run   <= '0';
        elsif run = '1' then
          count <= count + 1;
        end if;
      end if;
    end if;
  end process;

  cycles_o <= std_logic_vector(count);
end rtl;
