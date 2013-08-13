#!/usr/bin/perl
$device = "eth0";

#check if device has an expected ip address
$ifoutput = `ifconfig $device`;
if (!($ifoutput =~ /216.165.108/)) {
    $device = "eth1";
    $ifoutput = `ifconfig $device`;
    die "neither eth0 nor eth1 have right ip\n" if (!($ifoutput =~ /216.165.108/));
}
print "spreading interrupts for $device\n";

#figure out the list of irqs on $device
$irqoutput = `grep $device /proc/interrupts`;

@lines = split('\n', $irqoutput);
@irqs = ();
foreach $l (@lines) {
    @fields = split('\s+', $l);

    $num_cpu = 0;
    $active = 0;
    $irq = 0;
    for $f (@fields) {
        if ($f =~ /(\d+):/) {
            $irq = $1;
        }elsif ($f =~ /^(\d+)$/) {
            $cpu[$num_cpu] = $1;
            $num_cpu++;
            if ($1 > 100) {
                $active = 1;
            }
        }
    }
    if (!$total_cpus) {
        $total_cpus = $num_cpu;
    }else {
        die if ($total_cpus != $num_cpu);
    }
    if ($active) {
        push @irqs, $irq;
    }
}
print "irqs: @irqs\n";
print "total CPUs: $num_cpu\n";


#figure out how many CPU cores in total, which ones are hyper-threaded together
open FILE, "/proc/cpuinfo" or die "cannot open /proc/cpuinfo\n";
$virtual_id = 0;
$num_phy = 0;
$num_core = 0;
while (<FILE>) {
    if (/processor\s+:\s+(\d+)/) {
        $virtual_id = $1;
    }elsif (/physical id\s+:\s+(\d+)/) {
        $phy_id = $1;
        if (($phy_id+1) > $num_phy) {
            $num_phy = $phy_id+1;
        }
    }elsif (/core id\s+:\s+(\d+)/) {
        $core_id = $1;
        if (($core_id+1) > $num_core) {
            $num_core = $core_id+1;
        }
    }elsif (/cpu cores\s+:\s+(\d+)/) {
        print "processor $virtual_id: phy_id $phy_id core_id $core_id\n";
        $my_mask = 1 << $virtual_id;
        if (!defined($masks{"$phy_id.$core_id"})) {
            $masks{"$phy_id.$core_id"} = $my_mask;
        }else{
            $masks{"$phy_id.$core_id"} |= $my_mask;
        }
    }
}
close FILE;

for ($i = 0; $i < $num_phy; $i++) {
    for ($j = 0; $j < $num_core; $j++) {
        $key = "$i.$j";
        $mask = $masks{$key};
        print "key $key ";
        printf '%#x', $mask;
        print "\n";
    }
}

#spread irqs to differen physical cores
$phy_id = 0;
$core_id = 0;
for ($i = 0; $i <= $#irqs; $i++) {
    $key = "$phy_id.$core_id";
    $mask = sprintf("%x", $masks{$key});
    print "echo $mask > /proc/irq/$irqs[$i]/smp_affinity\n";
    `echo $mask > /proc/irq/$irqs[$i]/smp_affinity`;
    $core_id++;
    if ($core_id >= $num_core) {
        $phy_id++;
        $core_id = 0;
        die if ($phy_id == $num_phy);
    }
} 
