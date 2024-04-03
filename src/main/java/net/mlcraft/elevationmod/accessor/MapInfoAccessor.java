package net.mlcraft.elevationmod.accessor;

import net.mlcraft.elevationmod.world.gen.MLCraftMapInfo;
import net.minecraft.registry.entry.RegistryEntry;

public interface MapInfoAccessor {
    RegistryEntry<MLCraftMapInfo> mlcraft_getAMI();
    void mlcraft_setAMI(RegistryEntry<MLCraftMapInfo> ami);
}