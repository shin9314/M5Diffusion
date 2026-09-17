#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
int main() { @autoreleasepool {
 id<MTLDevice> d=MTLCreateSystemDefaultDevice();
 NSMutableDictionary *r=[@{@"name":d.name?:@"unavailable", @"unified_memory":@(d.hasUnifiedMemory),@"recommended_working_set":@(d.recommendedMaxWorkingSetSize),@"max_buffer_length":@(d.maxBufferLength),@"apple10":@([d supportsFamily:MTLGPUFamilyApple10]),@"metal4":@([d supportsFamily:MTLGPUFamilyMetal4])} mutableCopy];
 NSMutableDictionary *types=[NSMutableDictionary dictionary];
 NSArray *names=@[@"fp32",@"fp16",@"bf16",@"int8",@"int4"];
 MTLTensorDataType dt[]={MTLTensorDataTypeFloat32,MTLTensorDataTypeFloat16,MTLTensorDataTypeBFloat16,MTLTensorDataTypeInt8,MTLTensorDataTypeInt4};
 for(int i=0;i<5;i++){
   MTLTensorDescriptor *desc=[MTLTensorDescriptor new]; NSInteger dims[]={32,32};
   desc.dimensions=[[MTLTensorExtents alloc] initWithRank:2 values:dims]; desc.dataType=dt[i]; desc.storageMode=MTLStorageModeShared;
   NSError *err=nil; id<MTLTensor> t=[d newTensorWithDescriptor:desc error:&err];
   types[names[i]]=@{@"allocation_succeeded":@(t!=nil),@"error":err.localizedDescription?:@""};
 }
 r[@"tensor_storage_probe"]=types;
 r[@"fp8"] = @"SDK requires macOS 27; current OS 26.6: unavailable via this public tensor type";
 r[@"thermal_state"]=@([NSProcessInfo processInfo].thermalState);
 NSData *data=[NSJSONSerialization dataWithJSONObject:r options:NSJSONWritingPrettyPrinted error:nil]; puts([[NSString alloc]initWithData:data encoding:NSUTF8StringEncoding].UTF8String);
} }
