// probe_coopmat_props.c - enumerate the REAL cooperative-matrix configs this GPU supports
// (componentType, M, N, K, scope, subgroupSize). Never assume warp=32 or 16x16x16: query it.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

int main(void) {
    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.apiVersion = VK_API_VERSION_1_3;
    const char *iext[] = { VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME };
    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &app; ici.enabledExtensionCount = 1; ici.ppEnabledExtensionNames = iext;
    VkInstance inst; if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) { printf("instance fail\n"); return 1; }
    uint32_t n = 0; vkEnumeratePhysicalDevices(inst, &n, NULL);
    VkPhysicalDevice *pd = malloc(n * sizeof(*pd)); vkEnumeratePhysicalDevices(inst, &n, pd);
    printf("devices: %u\n", n);
    for (uint32_t d = 0; d < n; d++) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pd[d], &p);
        printf("\n[dev %u] %s  (api %u.%u.%u)\n", d, p.deviceName,
               VK_VERSION_MAJOR(p.apiVersion), VK_VERSION_MINOR(p.apiVersion), VK_VERSION_PATCH(p.apiVersion));
        uint32_t ec = 0; vkEnumerateDeviceExtensionProperties(pd[d], NULL, &ec, NULL);
        VkExtensionProperties *ex = malloc(ec * sizeof(*ex)); vkEnumerateDeviceExtensionProperties(pd[d], NULL, &ec, ex);
        int hasKHR = 0, hasNV = 0;
        for (uint32_t i = 0; i < ec; i++) {
            if (!strcmp(ex[i].extensionName, "VK_KHR_cooperative_matrix")) hasKHR = 1;
            if (!strcmp(ex[i].extensionName, "VK_NV_cooperative_matrix")) hasNV = 1;
        }
        printf("  coopmat ext: KHR=%d NV=%d\n", hasKHR, hasNV);
        if (!hasKHR) { free(ex); continue; }
        // create device with KHR coopmat
        float prio = 1.0f; VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
        qci.queueFamilyIndex = 0; qci.queueCount = 1; qci.pQueuePriorities = &prio;
        const char *dext[] = { "VK_KHR_cooperative_matrix" };
        VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = dext;
        VkDevice dev;
        if (vkCreateDevice(pd[d], &dci, NULL, &dev) != VK_SUCCESS) { printf("  device create fail\n"); free(ex); continue; }
        PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR g =
            (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
        if (!g) { printf("  no vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR\n"); vkDestroyDevice(dev, NULL); free(ex); continue; }
        uint32_t cnt = 0;
        if (g(pd[d], &cnt, NULL) != VK_SUCCESS) { printf("  query count fail\n"); vkDestroyDevice(dev, NULL); free(ex); continue; }
        VkCooperativeMatrixPropertiesKHR *pr = calloc(cnt, sizeof(*pr));
        for (uint32_t i = 0; i < cnt; i++) pr[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
        g(pd[d], &cnt, pr);
        printf("  %u coopmat configs:\n", cnt);
        for (uint32_t i = 0; i < cnt; i++) {
            printf("    Atype=%d Btype=%d Ctype=%d Rtype=%d  M=%u N=%u K=%u  scope=%d  saturatingAcc=%d\n",
                   pr[i].AType, pr[i].BType, pr[i].CType, pr[i].ResultType,
                   pr[i].MSize, pr[i].NSize, pr[i].KSize, pr[i].scope, pr[i].saturatingAccumulation);
        }
        free(pr); vkDestroyDevice(dev, NULL); free(ex);
    }
    vkDestroyInstance(inst, NULL); return 0;
}
